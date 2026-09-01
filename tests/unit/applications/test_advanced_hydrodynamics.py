#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _reference(shape=(4, 4, 3)):
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(shape[0], periodic=True),
            phx.discretization.UniformCellAxisSpec(shape[1], periodic=True),
            phx.discretization.UniformCellAxisSpec(shape[2], periodic=False),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, -1.0), (float(shape[0]), float(shape[1]), 0.0))))
    return phx.discretization.FiniteVolumePlan(
        grid, component_names=("hydrodynamics",)
    ).prepare()


def _hydrodynamics(*, surface_tension=0.0, wave=None):
    reference = _reference()
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        reference,
        jnp.full(reference.cell_shape[:2], -1.0),
        maximum_slope=0.8,
        maximum_iterations=100,
    )
    hydro = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        surface_tension=surface_tension,
        wave=wave,
        coupling_iterations=5,
        coupling_tolerance=1.0e-7,
    ).prepare()
    state = hydro.initial_state(jnp.zeros(reference.cell_shape[:2]))
    return hydro, state


def test_pressure_reference_is_invariant_to_common_offset():
    reference = _reference()
    surface = phx.applications.hydrodynamics.GraphSurfaceALEPlan(
        reference, jnp.full((4, 4), -1.0)
    )
    first = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        boundary=phx.applications.hydrodynamics.FreeSurfaceBoundaryPlan(
            gas_pressure=101325.0,
            reference_pressure=101325.0,
        ),
    ).prepare()
    second = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEPlan(
        surface,
        boundary=phx.applications.hydrodynamics.FreeSurfaceBoundaryPlan(
            gas_pressure=201325.0,
            reference_pressure=201325.0,
        ),
    ).prepare()
    eta = jnp.full((4, 4), 0.01)
    geometry = first.surface.geometry(0.0, eta, jnp.zeros_like(eta))

    first_stage = first.plan.boundary.stage(
        first.surface,
        geometry,
        eta,
        gravity=first.plan.gravity,
        density=first.plan.density,
    )
    second_stage = second.plan.boundary.stage(
        second.surface,
        geometry,
        eta,
        gravity=second.plan.gravity,
        density=second.plan.density,
    )

    np.testing.assert_allclose(
        first_stage.surface_pressure_head,
        second_stage.surface_pressure_head,
    )


def test_variational_capillarity_flat_and_finite_difference():
    hydro, _ = _hydrodynamics(surface_tension=0.072)
    eta = jnp.zeros((4, 4))
    flat = hydro.capillarity.evaluate(eta, hydro.plan.density)
    perturbation = jnp.zeros_like(eta).at[1, 1].set(1.0)
    epsilon = 1.0e-6
    numerical = (
        hydro.capillarity.surface_area(eta + epsilon * perturbation)
        - hydro.capillarity.surface_area(eta - epsilon * perturbation)
    ) / (2.0 * epsilon)
    analytic = jnp.vdot(flat.generalized_force / 0.072, perturbation)

    assert bool(flat.successful)
    np.testing.assert_allclose(flat.generalized_force, 0.0, atol=1e-12)
    np.testing.assert_allclose(numerical, analytic, rtol=5e-5, atol=1e-8)
    assert flat.timestep_limit > 0.0


def test_incident_wave_provider_is_phase_coherent_and_restartable():
    component = phx.equations.WaveComponent(0.02, 2.0, 0.3, 0.7)
    provider = phx.equations.IncidentWavePlan((component,), 2.0, ramp_time=1.0)
    coordinates = jnp.asarray(((0.5, 0.25, -0.5),))

    first = provider.sample(0.5, coordinates)
    second = provider.sample(0.5, coordinates)

    assert bool(first.valid)
    assert provider.components[0].wavenumber > 0.0
    np.testing.assert_allclose(first.eta, second.eta)
    np.testing.assert_allclose(first.velocity, second.velocity)
    np.testing.assert_allclose(first.pressure_head, second.pressure_head)


def test_wave_forcing_and_active_absorption_are_finite():
    provider = phx.equations.IncidentWavePlan(
        (phx.equations.WaveComponent(0.01, 1.5),), 1.0
    )
    weights = jnp.zeros((4, 4, 3)).at[:2].set(1.0)
    sponge = jnp.zeros_like(weights).at[-2:].set(0.5)
    wave = phx.applications.hydrodynamics.WaveForcingPlan(
        provider,
        weights,
        sponge,
        active_gain=0.2,
    )
    hydro, state = _hydrodynamics(wave=wave)
    view = hydro.view(state)
    controller = wave.initial_controller_state(state.eta.shape, state.eta.dtype)

    result = wave.evaluate(
        hydro.surface,
        view.geometry,
        0.1,
        view.velocity,
        state.eta,
        controller,
    )
    updated = wave.update_controller(
        controller, 0.1, state.eta, result.sample.eta[..., 0]
    )
    diagnostics = wave.diagnostics(updated)

    assert bool(result.valid)
    assert bool(diagnostics.valid)
    assert jnp.all(jnp.isfinite(result.eta_rate_source))


def test_vertical_rezone_preserves_scalar_content_and_shoreline_handoff():
    hydro, state = _hydrodynamics()
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )
    rezone = phx.applications.hydrodynamics.FreeSurfaceRezonePlan(1.4)

    result = rezone.rezone(hydro, continuation)
    shoreline = phx.applications.hydrodynamics.GraphShorelineEventPlan(
        rezone_height=0.1,
        dry_height=0.01,
    ).evaluate(hydro, continuation)

    assert bool(result.evidence.conservative)
    assert bool(result.evidence.finite)
    assert int(result.state.mesh_epoch) == 1
    assert shoreline.status in ("continue", "rezone")


def test_capillary_wave_step_closes_and_updates_controller():
    provider = phx.equations.IncidentWavePlan(
        (phx.equations.WaveComponent(1.0e-4, 1.0),), 1.0
    )
    wave = phx.applications.hydrodynamics.WaveForcingPlan(
        provider,
        jnp.zeros((4, 4, 3)).at[:2].set(0.25),
        jnp.zeros((4, 4, 3)).at[-2:].set(0.1),
        active_gain=0.1,
    )
    hydro, state = _hydrodynamics(surface_tension=0.072, wave=wave)
    continuation = (
        phx.applications.hydrodynamics.FreeSurfaceALEContinuationState.initialize(state)
    )

    result = phx.applications.hydrodynamics.OnePhaseFreeSurfaceALEMethod(hydro).step(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        continuation,
        jnp.asarray(1.0e-4),
        None,
    )

    assert bool(result.successful)
    assert result.accepted_state.wave_controller is not None
    assert jnp.isfinite(result.accepted_state.ledger.surface_energy_change)
    assert jnp.isfinite(result.accepted_state.ledger.wave_work)
    assert jnp.isfinite(result.accepted_state.ledger.sponge_dissipation)
