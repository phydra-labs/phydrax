#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.interfacial_transport as it
from tests._support.film_meshes import icosphere, obtuse_mesh, planar_grid


_STEP = eqx.filter_jit(lambda prepared, state, step_size: prepared.step(state, step_size))


def _supported_plan(
    surface: it.PreparedFilmSurface,
    *,
    boundary_policy: it.FilmBoundaryPolicy = "no-flux",
    boundary_thickness_m: float = 0.0,
    tolerance: float = 1e-10,
    maximum_iterations: int = 40,
) -> it.PreparedSurfaceLubrication:
    return it.SurfaceLubricationPlan(
        surface,
        mobility_law="one-sided-substrate",
        surface_tension_n_m=1.0,
        viscosity_pa_s=1.0,
        boundary_policy=boundary_policy,
        boundary_thickness_m=boundary_thickness_m,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
    ).prepare()


def _mode_amplitude(
    prepared: it.PreparedSurfaceLubrication,
    state: it.SurfaceLubricationState,
    mode: np.ndarray,
) -> float:
    area = np.asarray(prepared.plan.surface.vertex_area)
    thickness = np.asarray(prepared.thickness(state))
    mean = np.sum(area * thickness) / np.sum(area)
    return float(np.sum(area * (thickness - mean) * mode) / np.sum(area * mode**2))


@pytest.fixture(scope="module")
def planar() -> it.PreparedSurfaceLubrication:
    return _supported_plan(it.prepare_film_surface(planar_grid(20, 20)))


def test_uniform_planar_film_is_invariant_and_has_zero_capillary_pressure(
    planar: it.PreparedSurfaceLubrication,
) -> None:
    surface = planar.plan.surface
    pressure = it.film_capillary_pressure(
        surface,
        jnp.full((surface.topology.num_vertices,), 0.1),
        0.07,
        configuration="symmetric-free",
    )
    assert float(jnp.max(jnp.abs(pressure))) < 1e-12
    state = planar.initial_state(0.1)
    result = _STEP(planar, state, 1.0)
    assert int(result.status) == it.FilmStepStatus.ACCEPTED
    np.testing.assert_allclose(
        result.state.liquid_volume_m3, state.liquid_volume_m3, rtol=1e-11
    )


def test_planar_capillary_levelling_matches_fourier_decay_rate(
    planar: it.PreparedSurfaceLubrication,
) -> None:
    x = np.asarray(planar.plan.surface.coordinates[:, 0])
    wavenumber = 2.0 * np.pi
    mode = np.cos(wavenumber * x)
    thickness, step_size = 0.1, 0.02
    state = planar.initial_state(thickness * (1.0 + 1e-5 * mode))
    result = _STEP(planar, state, step_size)
    ratio = _mode_amplitude(planar, result.state, mode) / _mode_amplitude(
        planar, state, mode
    )
    spacing = 1.0 / 20
    eigenvalue = 2.0 * (1.0 - np.cos(wavenumber * spacing)) / spacing**2
    discrete_rate = thickness**3 * eigenvalue**2 / 3.0
    np.testing.assert_allclose(ratio, 1.0 / (1.0 + discrete_rate * step_size), rtol=1e-6)
    continuum_rate = thickness**3 * wavenumber**4 / 3.0
    np.testing.assert_allclose(-np.log(ratio) / step_size, continuum_rate, rtol=0.03)


def test_spherical_harmonic_decay_on_certified_mesh() -> None:
    surface = it.prepare_film_surface(icosphere(3))
    assert bool(surface.evidence.conductance_admissible)
    prepared = _supported_plan(surface)
    z = np.asarray(surface.coordinates[:, 2])
    thickness, step_size = 0.1, 0.05
    quadrupole = 3.0 * z**2 - 1.0
    state = prepared.initial_state(thickness * (1.0 + 1e-4 * quadrupole))
    result = _STEP(prepared, state, step_size)
    ratio = _mode_amplitude(prepared, result.state, quadrupole) / _mode_amplitude(
        prepared, state, quadrupole
    )
    # s_l = sigma h^3 lambda (lambda - 2/R^2) / (3 mu), lambda = l (l + 1) / R^2.
    np.testing.assert_allclose(
        (1.0 / ratio - 1.0) / step_size, 8.0 * thickness**3, rtol=0.03
    )
    dipole_state = prepared.initial_state(thickness * (1.0 + 1e-4 * z))
    dipole = _STEP(prepared, dipole_state, step_size)
    dipole_ratio = _mode_amplitude(prepared, dipole.state, z) / _mode_amplitude(
        prepared, dipole_state, z
    )
    assert abs(dipole_ratio - 1.0) < 0.02 * (8.0 * thickness**3 * step_size)


def test_closed_surface_drainage_conserves_volume_and_dissipates_energy(
    planar: it.PreparedSurfaceLubrication,
) -> None:
    rng = np.random.default_rng(3)
    count = planar.plan.surface.topology.num_vertices
    state = planar.initial_state(0.1 * (1.0 + 0.1 * rng.uniform(-1.0, 1.0, count)))
    total = float(jnp.sum(state.liquid_volume_m3))
    for _ in range(3):
        result = _STEP(planar, state, 0.05)
        assert int(result.status) == it.FilmStepStatus.ACCEPTED
        evidence = result.evidence
        assert bool(evidence.positivity_guaranteed)
        assert bool(evidence.dissipation_guaranteed)
        assert float(evidence.energy_change_j) < 0.0
        assert float(result.entropy_change) < 0.0
        state = result.state
    assert abs(float(jnp.sum(state.liquid_volume_m3)) - total) <= 1e-14 * total


def test_fixed_thickness_boundary_reports_reservoir_exchange() -> None:
    surface = it.prepare_film_surface(planar_grid(12, 12))
    prepared = _supported_plan(
        surface, boundary_policy="fixed-thickness", boundary_thickness_m=0.05
    )
    state = prepared.initial_state(0.1)
    result = _STEP(prepared, state, 0.5)
    assert int(result.status) == it.FilmStepStatus.ACCEPTED
    boundary = np.asarray(surface.topology.boundary_vertices)
    np.testing.assert_allclose(
        np.asarray(prepared.thickness(result.state))[boundary], 0.05
    )
    change = float(jnp.sum(result.state.liquid_volume_m3 - state.liquid_volume_m3))
    assert float(result.evidence.boundary_exchange_m3) < 0.0
    assert abs(change - float(result.evidence.boundary_exchange_m3)) < 1e-15
    np.testing.assert_allclose(
        np.sum(np.asarray(result.boundary_exchange_m3)),
        result.evidence.boundary_exchange_m3,
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_array_equal(np.asarray(result.boundary_exchange_m3)[~boundary], 0.0)
    assert not bool(result.evidence.dissipation_guaranteed)


def test_inadmissible_conductance_rejects_candidate() -> None:
    surface = it.prepare_film_surface(obtuse_mesh())
    assert not bool(surface.evidence.conductance_admissible)
    assert float(surface.evidence.minimum_edge_conductance) < 0.0
    prepared = _supported_plan(surface)
    state = prepared.initial_state(jnp.asarray((0.1, 0.2, 0.1, 0.2)))
    result = _STEP(prepared, state, 0.1)
    assert int(result.status) == it.FilmStepStatus.INADMISSIBLE_CONDUCTANCE
    assert not bool(result.evidence.positivity_guaranteed)
    np.testing.assert_array_equal(result.state.liquid_volume_m3, state.liquid_volume_m3)


def test_unconverged_solve_rejects_candidate() -> None:
    surface = it.prepare_film_surface(planar_grid(8, 8))
    prepared = _supported_plan(surface, maximum_iterations=1, tolerance=1e-14)
    x = np.asarray(surface.coordinates[:, 0])
    state = prepared.initial_state(0.1 * (1.0 + 0.5 * np.cos(2.0 * np.pi * x)))
    result = _STEP(prepared, state, 5.0)
    assert int(result.status) == it.FilmStepStatus.SOLVE_FAILED
    assert not bool(result.evidence.converged)
    np.testing.assert_array_equal(result.state.liquid_volume_m3, state.liquid_volume_m3)


def test_large_nonlinear_drainage_step_converges_with_accurate_inner_solves() -> None:
    # 30 % random thickness at dt ~ 1e3 decay times: adaptive forcing stalled
    # the trust region for 40 steps here; accurate inner solves need about 8.
    surface = it.prepare_film_surface(planar_grid(6, 6))
    prepared = _supported_plan(surface)
    rng = np.random.default_rng(3)
    count = surface.topology.num_vertices
    state = prepared.initial_state(0.1 * (1.0 + 0.3 * rng.uniform(-1.0, 1.0, count)))
    result = _STEP(prepared, state, 0.5)
    assert int(result.status) == it.FilmStepStatus.ACCEPTED
    assert bool(result.evidence.converged)
    assert int(result.evidence.nonlinear_iterations) <= 12
    total = float(jnp.sum(state.liquid_volume_m3))
    assert abs(float(jnp.sum(result.state.liquid_volume_m3)) - total) <= 1e-14 * total


def test_attractive_van_der_waals_thins_a_depression_and_flags_rupture() -> None:
    surface = it.prepare_film_surface(planar_grid(16, 16, 1e-5, 1e-5))
    prepared = it.SurfaceLubricationPlan(
        surface,
        mobility_law="one-sided-substrate",
        surface_tension_n_m=0.04,
        viscosity_pa_s=1e-3,
        disjoining=it.VanDerWaalsDisjoiningPressure(1e-19),
        rupture_thickness_m=9.5e-9,
    ).prepare()
    x = np.asarray(surface.coordinates[:, 0])
    state = prepared.initial_state(1e-8 * (1.0 - 0.05 * np.cos(np.pi * x / 1e-5)))
    initial_minimum = float(jnp.min(prepared.thickness(state)))
    result = _STEP(prepared, state, 1e-3)
    assert int(result.status) == it.FilmStepStatus.ACCEPTED
    assert not bool(result.evidence.dissipation_guaranteed)
    assert float(result.evidence.minimum_thickness_m) < initial_minimum
    assert bool(jnp.any(result.evidence.rupture_mask))
