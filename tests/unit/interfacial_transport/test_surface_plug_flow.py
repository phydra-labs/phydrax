#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax.interfacial_transport as it
from phydrax.nonlinear import NonlinearStatus
from tests._support.film_meshes import icosphere, planar_grid


_CAPACITY = 4e-6
_STEP = eqx.filter_jit(lambda prepared, state, step_size: prepared.step(state, step_size))


def _law() -> it.LangmuirSurfactantLaw:
    return it.LangmuirSurfactantLaw(0.072, 298.15, _CAPACITY)


def test_marangoni_wave_speed_uses_twice_the_single_interface_elasticity() -> None:
    length = 1e-2
    surface = it.prepare_film_surface(planar_grid(64, 2, length, length / 32))
    law = _law()
    prepared = it.SurfacePlugFlowPlan(surface, law, density_kg_m3=1000.0).prepare()
    x = np.asarray(surface.coordinates[:, 0])
    wavenumber = 2.0 * np.pi / length
    mode = np.cos(wavenumber * x)
    thickness, concentration = 1e-6, 2e-6
    speed = np.sqrt(
        2.0 * float(law.gibbs_elasticity(concentration)) / (1000.0 * thickness)
    )
    period = 2.0 * np.pi / (speed * wavenumber)
    step_size = period / 200.0
    state = prepared.initial_state(thickness, concentration * (1.0 + 1e-4 * mode))
    area = np.asarray(surface.vertex_area)

    def amplitude(current: it.SurfacePlugFlowState) -> float:
        density = np.asarray(current.surfactant_amount_mol) / area - concentration
        return float(np.sum(area * density * mode) / np.sum(area * mode**2))

    previous = amplitude(state)
    for index in range(1, 120):
        result = _STEP(prepared, state, step_size)
        assert int(result.status) == it.FilmStepStatus.ACCEPTED
        state = result.state
        current = amplitude(state)
        if current < 0.0:
            crossing = (index - 1 + previous / (previous - current)) * step_size
            break
        previous = current
    else:
        raise AssertionError("The Marangoni wave did not reach its first node.")
    np.testing.assert_allclose(crossing, 0.25 * period, rtol=0.03)
    assert float(result.plug_flow.marangoni_courant_number) > 0.1


def test_force_free_transport_conserves_momentum_surfactant_and_volume() -> None:
    surface = it.prepare_film_surface(planar_grid(16, 16, 1e-3, 1e-3))
    prepared = it.SurfacePlugFlowPlan(
        surface, _law(), density_kg_m3=1000.0, viscosity_pa_s=1e-3
    ).prepare()
    points = np.asarray(surface.coordinates)
    bump = np.exp(-(((points[:, :2] - 5e-4) / 1.5e-4) ** 2).sum(axis=1))
    velocity = np.stack((1e-2 * bump, 5e-3 * bump, np.zeros_like(bump)), axis=1)
    # A clean film has uniform tension, so no Marangoni force acts.
    state = prepared.initial_state(1e-6 * (1.0 + 0.2 * bump), 0.0, velocity)
    momentum = np.asarray(jnp.sum(state.momentum_kg_m_s, axis=0))
    surfactant = float(state.total_surfactant_mol())
    volume = float(jnp.sum(state.liquid_volume_m3))
    kinetic = []
    for _ in range(4):
        result = _STEP(prepared, state, 1e-4)
        assert int(result.status) == it.FilmStepStatus.ACCEPTED
        assert float(result.plug_flow.viscous_dissipation_w) >= 0.0
        kinetic.append(float(result.plug_flow.kinetic_energy_change_j))
        state = result.state
    np.testing.assert_allclose(
        np.asarray(jnp.sum(state.momentum_kg_m_s, axis=0)),
        momentum,
        rtol=0.0,
        atol=1e-9 * np.abs(momentum).max(),
    )
    assert float(state.total_surfactant_mol()) == surfactant
    np.testing.assert_allclose(float(jnp.sum(state.liquid_volume_m3)), volume, rtol=1e-14)
    assert all(change < 0.0 for change in kinetic)


def test_gravity_and_air_drag_impulse_balance_momentum() -> None:
    surface = it.prepare_film_surface(planar_grid(8, 8, 1e-3, 1e-3))
    prepared = it.SurfacePlugFlowPlan(
        surface,
        _law(),
        density_kg_m3=1000.0,
        viscosity_pa_s=1e-3,
        surface_shear_viscosity_n_s_m=1e-7,
        surface_dilatational_viscosity_n_s_m=1e-7,
        air_drag_coefficient_kg_m2_s=0.05,
        gravity_m_s2=(-9.81, 0.0, 0.0),
    ).prepare()
    state = prepared.initial_state(1e-6, 0.0, (0.0, 1e-2, 0.0))
    result = _STEP(prepared, state, 1e-4)
    assert int(result.status) == it.FilmStepStatus.ACCEPTED
    evidence = result.plug_flow
    np.testing.assert_allclose(
        evidence.momentum_change_n_s,
        evidence.external_impulse_n_s,
        rtol=1e-10,
        atol=1e-12 * float(jnp.max(jnp.abs(evidence.external_impulse_n_s))),
    )
    assert float(evidence.drag_dissipation_w) > 0.0
    assert float(result.evidence.minimum_thickness_m) > 0.0


def test_finite_normal_momentum_is_rejected_without_projection() -> None:
    surface = it.prepare_film_surface(planar_grid(4, 4, 1e-3, 1e-3))
    prepared = it.SurfacePlugFlowPlan(
        surface, _law(), density_kg_m3=1000.0
    ).prepare()
    state = prepared.initial_state(1e-6, 1e-6)
    normal_momentum = 1e-12 * surface.vertex_normal
    invalid = it.SurfacePlugFlowState(
        state.liquid_volume_m3,
        state.surfactant_amount_mol,
        None,
        state.momentum_kg_m_s + normal_momentum,
        topology_id=state.topology_id,
        geometry_revision=state.geometry_revision,
    )

    result = _STEP(prepared, invalid, 1e-4)

    assert int(result.status) == it.FilmStepStatus.INADMISSIBLE_INPUT
    assert not bool(result.plug_flow.tangential_momentum_admissible)
    assert (
        float(result.plug_flow.tangential_momentum_residual_kg_m_s)
        > float(result.plug_flow.tangential_momentum_tolerance_kg_m_s)
    )
    np.testing.assert_array_equal(
        result.state.momentum_kg_m_s, invalid.momentum_kg_m_s
    )


def test_insoluble_surfactant_is_conserved_under_marangoni_flow() -> None:
    surface = it.prepare_film_surface(planar_grid(12, 12, 1e-3, 1e-3))
    prepared = it.SurfacePlugFlowPlan(
        surface, _law(), density_kg_m3=1000.0, viscosity_pa_s=1e-3
    ).prepare()
    x = np.asarray(surface.coordinates[:, 0])
    state = prepared.initial_state(1e-6, 2e-6 * (1.0 + 0.1 * np.cos(np.pi * x / 1e-3)))
    total = float(state.total_surfactant_mol())
    for _ in range(3):
        result = _STEP(prepared, state, 1e-4)
        assert int(result.status) == it.FilmStepStatus.ACCEPTED
        assert float(result.plug_flow.viscous_dissipation_w) >= 0.0
        state = result.state
    np.testing.assert_allclose(float(state.total_surfactant_mol()), total, rtol=1e-13)


def test_implicit_stage_rejects_nonpositive_tension_atomically() -> None:
    surface = it.prepare_film_surface(planar_grid(4, 4, 1e-3, 1e-3))
    law = _law()
    zero_tension = _CAPACITY * (
        1.0 - np.exp(-float(law.clean_surface_tension_n_m / law.surface_pressure_scale_n_m))
    )
    concentration = 0.5 * (zero_tension + _CAPACITY)
    prepared = it.SurfacePlugFlowPlan(
        surface, law, density_kg_m3=1000.0
    ).prepare()
    state = prepared.initial_state(1e-6, concentration)
    result = _STEP(prepared, state, 1e-6)
    assert int(result.status) == it.FilmStepStatus.NONPOSITIVE_TENSION
    assert float(result.plug_flow.maximum_coverage) < 1.0
    assert float(result.plug_flow.minimum_surface_tension_n_m) < 0.0
    assert not bool(result.plug_flow.surface_state_admissible)
    assert (
        int(result.plug_flow.terminal_nonlinear_stage)
        == it.PlugFlowNonlinearStage.SURFACTANT
    )
    assert bool(result.evidence.converged)
    assert int(result.evidence.nonlinear_iterations) == 1
    np.testing.assert_array_equal(
        result.state.surfactant_amount_mol, state.surfactant_amount_mol
    )
    np.testing.assert_array_equal(
        result.state.momentum_kg_m_s, state.momentum_kg_m_s
    )


def test_soluble_exchange_failure_owns_terminal_nonlinear_evidence() -> None:
    surface = it.prepare_film_surface(planar_grid(4, 4, 1e-3, 1e-3))
    kinetics = it.AdsorptionKinetics(1e-5, 1.0, _CAPACITY)
    prepared = it.SurfacePlugFlowPlan(
        surface,
        _law(),
        density_kg_m3=1000.0,
        kinetics=kinetics,
        maximum_iterations=1,
    ).prepare()
    state = prepared.initial_state(1e-6, 1e-7, (0.0, 0.0, 0.0), 1.0)

    result = _STEP(prepared, state, 1e-3)

    assert int(result.status) == it.FilmStepStatus.SOLVE_FAILED
    assert (
        int(result.plug_flow.terminal_nonlinear_stage)
        == it.PlugFlowNonlinearStage.SURFACTANT
    )
    assert (
        int(result.evidence.nonlinear_status)
        == NonlinearStatus.MAXIMUM_STEPS_REACHED
    )
    assert int(result.evidence.nonlinear_iterations) == 1
    assert not bool(result.evidence.converged)
    np.testing.assert_array_equal(
        result.state.surfactant_amount_mol, state.surfactant_amount_mol
    )


def _rest_profile(
    height: np.ndarray, area: np.ndarray, volume: float, ratio: float
) -> np.ndarray:
    """Rest thickness of the insoluble non-diffusive Langmuir film under gravity.

    ``Gamma = ratio * h`` and ``2 grad sigma + rho h g_t = 0`` give
    ``logit(Gamma / Gamma_inf) = C - rho g z / (2 R T ratio)`` (the Langmuir form
    of Huang et al. 2020, eq. 36); ``C`` is fixed by the liquid volume.
    """
    slope = 1000.0 * 9.81 / (2.0 * 8.314462618 * 298.15 * ratio)

    def thickness(offset: float) -> np.ndarray:
        return _CAPACITY / ratio / (1.0 + np.exp(slope * height - offset))

    low, high = -80.0, 20.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        low, high = (
            (low, middle)
            if np.sum(area * thickness(middle)) > volume
            else (middle, high)
        )
    return thickness(0.5 * (low + high))


def test_fixed_sphere_relaxes_to_the_gravity_marangoni_rest_profile() -> None:
    radius, thickness, drag = 0.02, 1e-6, 0.05
    # Gamma_0 makes the rest exponent rho g R h_0 / (2 R T Gamma_0) equal one.
    concentration = 1000.0 * 9.81 * radius * thickness / (2.0 * 8.314462618 * 298.15)
    mesh = icosphere(1, radius)
    surface = it.prepare_film_surface(mesh)
    prepared = it.SurfacePlugFlowPlan(
        surface,
        _law(),
        density_kg_m3=1000.0,
        air_drag_coefficient_kg_m2_s=drag,
        gravity_m_s2=(0.0, 0.0, -9.81),
    ).prepare()
    state = prepared.initial_state(thickness, concentration)
    area = np.asarray(surface.vertex_area)
    height = np.asarray(surface.coordinates[:, 2])
    volume = float(jnp.sum(state.liquid_volume_m3))
    surfactant = float(state.total_surfactant_mol())
    rest = _rest_profile(height, area, volume, concentration / thickness)
    speeds = []
    for index in range(120):
        result = _STEP(prepared, state, 5e-3)
        assert int(result.status) == it.FilmStepStatus.ACCEPTED
        state = result.state
        if index in (59, 119):
            velocity = prepared.velocity(state)
            speeds.append(float(jnp.max(jnp.linalg.norm(velocity, axis=1))))
    film = np.asarray(state.liquid_volume_m3) / area
    uniform_error = np.sqrt(
        np.sum(area * (thickness - rest) ** 2) / np.sum(area * rest**2)
    )
    error = np.sqrt(np.sum(area * (film - rest) ** 2) / np.sum(area * rest**2))
    # The uniform start is 48 % from the rest profile; 42 vertices resolve it
    # to about 5 % (1.4 % at 162 vertices in the qualification campaign).
    assert uniform_error > 0.4
    assert error < 0.07
    top, bottom = int(np.argmax(height)), int(np.argmin(height))
    np.testing.assert_allclose(
        film[top] / film[bottom], rest[top] / rest[bottom], rtol=0.1
    )
    # The film is at rest up to a steady discretization-level circulation.
    drainage_speed = 1000.0 * thickness * 9.81 / drag
    assert speeds[-1] < 0.05 * drainage_speed
    assert abs(speeds[-1] - speeds[0]) < 0.1 * speeds[-1]
    np.testing.assert_allclose(
        float(jnp.sum(state.liquid_volume_m3)), volume, rtol=1e-13
    )
    np.testing.assert_allclose(
        float(state.total_surfactant_mol()), surfactant, rtol=1e-13
    )
