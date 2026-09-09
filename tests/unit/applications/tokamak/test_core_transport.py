import math

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _equilibrium():
    count = 65
    r = np.linspace(1.0, 3.0, count)
    z = np.linspace(-1.0, 1.0, count)
    rr, zz = np.meshgrid(r, z)
    minor_radius = 0.8
    psi = ((rr - 2.0) ** 2 + zz**2) / minor_radius**2
    theta = 2.0 * math.pi * np.arange(128) / 128
    boundary = np.stack(
        (2.0 + minor_radius * np.cos(theta), minor_radius * np.sin(theta)), axis=-1
    )
    profile = np.linspace(0.0, 1.0, count)
    return phx.applications.tokamak.AxisymmetricEquilibrium(
        phx.applications.tokamak.AxisymmetricMachineFrame("analytic-circular-machine"),
        phx.applications.tokamak.TokamakMagneticConvention.canonical(),
        r,
        z,
        psi,
        np.full(count, 10.0),
        1.0e5 * (1.0 - profile),
        np.zeros(count),
        -1.0e5 * np.ones(count),
        1.0 + profile,
        boundary,
        boundary,
        np.asarray([2.0, 0.0]),
        0.0,
        1.0,
        2.0,
        5.0,
        1.0e6,
        "analytic-circular-equilibrium",
    )


def _geometry():
    return phx.applications.tokamak.FluxSurfacePlan(
        np.asarray([0.0, 0.25, 0.5, 0.75]),
        poloidal_count=64,
        radial_search_count=128,
    ).prepare(_equilibrium())


def test_flux_surface_geometry_recovers_circular_torus_measures():
    geometry = _geometry()
    rho = geometry.rho_faces
    expected_volume = 2.0 * math.pi**2 * 2.0 * (0.8 * rho) ** 2
    expected_area = 4.0 * math.pi**2 * 2.0 * 0.8 * rho
    np.testing.assert_allclose(geometry.enclosed_volume_m3, expected_volume, rtol=0.015)
    np.testing.assert_allclose(
        geometry.surface_area_m2[1:], expected_area[1:], rtol=0.015
    )
    assert bool(geometry.prepare().evidence.successful)


def test_core_transport_conserves_particles_and_energy_without_boundary_flux():
    geometry = _geometry()
    prepared = phx.applications.tokamak.TokamakCoreTransportPlan(geometry, 1.0).prepare()
    cell_count = prepared.cell_count
    thermal_energy = float(
        phx.units.conversion_factor(phx.units.KILOELECTRONVOLT, phx.units.JOULE)
    )
    state = phx.applications.tokamak.TokamakCoreState(
        np.linspace(1.0e19, 0.5e19, cell_count),
        np.full(cell_count, 5.0 * thermal_energy),
        np.full(cell_count, 4.0 * thermal_energy),
    )
    conductance = np.zeros((cell_count + 1,))
    conductance[1:-1] = 0.1
    coefficients = phx.applications.tokamak.TokamakTransportCoefficients(
        conductance, conductance, conductance
    )
    sources = phx.applications.tokamak.TokamakTransportSources(
        np.zeros(cell_count), np.zeros(cell_count), np.zeros(cell_count)
    )

    result = prepared.step(state, 0.01, coefficients, sources)

    assert bool(result.successful)
    np.testing.assert_allclose(
        result.ledger.candidate_totals,
        result.ledger.initial_totals,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert bool(result.ledger.successful)
    assert (
        np.max(
            np.abs(result.ledger.closure_residual)
            / np.maximum(1.0, np.abs(result.ledger.initial_totals))
        )
        < 1.0e-12
    )


def test_core_transport_is_differentiable_through_native_line_solve():
    prepared = phx.applications.tokamak.TokamakCoreTransportPlan(
        _geometry(), 1.0
    ).prepare()
    count = prepared.cell_count
    state = phx.applications.tokamak.TokamakCoreState(
        jnp.linspace(2.0, 1.0, count),
        jnp.ones(count),
        0.8 * jnp.ones(count),
    )
    sources = phx.applications.tokamak.TokamakTransportSources(
        jnp.zeros(count), jnp.zeros(count), jnp.zeros(count)
    )

    def objective(interior_conductance):
        faces = jnp.zeros((count + 1,)).at[1:-1].set(interior_conductance)
        coefficients = phx.applications.tokamak.TokamakTransportCoefficients(
            faces, faces, faces
        )
        result = prepared.step(state, 0.05, coefficients, sources)
        return result.accepted_state.electron_density_m3[-1]

    derivative = jax.grad(objective)(jnp.asarray(0.2))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0
