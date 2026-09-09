# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.atmosphere._equilibration import (
    global_flux_residuals,
    precondition_global_fluxes,
)
from phydrax.applications.atmosphere._global import (
    GlobalPrimitiveEquationPlan,
    read_global_atmosphere_checkpoint,
    write_global_atmosphere_checkpoint,
)
from phydrax.applications.atmosphere._global_surface import GlobalSurfacePhysics
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.atmosphere._processes import GlobalAtmosphereProcesses
from phydrax.applications.atmosphere._radiation import (
    ColumnOpticalProperties,
    ColumnRadiationPlan,
)
from phydrax.applications.atmosphere._surface import BulkSurfaceExchangePlan, WetSlabPlan
from phydrax.applications.geophysics._vertical import HybridPressureCoordinate
from phydrax.discretization.spectral._spherical import SphericalSpectralPlan
from tools.global_feedback_qualification import make_model


@pytest.fixture(scope="module")
def space():
    return SphericalSpectralPlan(3, sampling="gl").prepare(radius=6.371e6)


def boundary(*, capacity=2e6, solar=1361.0, p2=0.0):
    thermo = MoistThermodynamicPlan()
    optics = ColumnOpticalProperties(
        shortwave_absorption=(1e-5, 0.001, 0.01, 0.01),
        shortwave_scattering=(0.0, 0.0, 0.0, 0.0),
        longwave_absorption=(1e-4, 0.01, 0.01, 0.01),
        reference_id="synthetic-global-surface-regression",
    )
    return GlobalSurfacePhysics(
        WetSlabPlan(thermo, dry_heat_capacity=capacity),
        BulkSurfaceExchangePlan(stability="neutral"),
        ColumnRadiationPlan(optics),
        solar_constant=solar,
        solar_p2=p2,
    )


def model_at(space, surface):
    return GlobalPrimitiveEquationPlan(
        space,
        HybridPressureCoordinate([0.1, 0.0], [0.0, 1.0]),
        dt=20.0,
        rotation_rate=0.0,
        processes=GlobalAtmosphereProcesses(
            thermodynamics=surface.slab.thermodynamics,
            surface_physics=surface,
            cadence=2,
        ),
    ).prepare()


def initialize(model, *, water=1000.0, temperature=290.0):
    return model.initialize(
        temperature=280.0,
        vapor=0.002,
        liquid=1e-6,
        ice=1e-6,
        surface_temperature=temperature,
        surface_water=water,
    )


def test_current_slab_controls_both_water_and_heat_and_pressure(space):
    model = model_at(space, boundary())
    cold, warm = (
        initialize(model, temperature=280.0),
        initialize(model, temperature=295.0),
    )
    evaluate = eqx.filter_jit(model.advance)
    cold_result, warm_result = evaluate(cold), evaluate(warm)
    assert bool(cold_result.evidence.accepted) and bool(warm_result.evidence.accepted)
    cold_view = model.view(cold_result.continuation.state)
    warm_view = model.view(warm_result.continuation.state)
    assert np.all(np.asarray(warm_view.temperature) > np.asarray(cold_view.temperature))
    assert np.all(
        np.asarray(warm_view.surface_pressure) > np.asarray(cold_view.surface_pressure)
    )
    assert np.all(
        np.asarray(warm_result.continuation.state.surface_water)
        < np.asarray(cold_result.continuation.state.surface_water)
    )
    for initial, result in ((cold, cold_result), (warm, warm_result)):
        before, after = (
            model.inventories(initial.state),
            model.inventories(result.continuation.state),
        )
        np.testing.assert_allclose(after[:2], before[:2], rtol=2e-13)


def test_solar_flux_is_latitude_dependent_normalized_and_radiatively_closed(space):
    model = model_at(space, boundary(p2=-0.48))
    initial = initialize(model)
    solar = initial.held_forcing.solar_down
    area = 4 * np.pi * space.radius**2
    np.testing.assert_allclose(
        model.work_space.integral(solar) / area, 1361 / 4, rtol=2e-14
    )
    theta = np.asarray(model.work_space.transform.theta)
    assert np.mean(np.asarray(solar)[np.argmin(np.abs(theta - np.pi / 2))]) > np.mean(
        np.asarray(solar)[np.argmin(theta)]
    )
    flux = model.plan.processes.surface_physics.evaluate(
        model.plan.processes.thermodynamics,
        model.view(initial.state),
        initial.state.surface_water,
        initial.state.surface_energy,
        solar,
    )
    assert np.all(flux.successful)
    np.testing.assert_allclose(
        jnp.sum(flux.radiation.heating, axis=-1)
        + flux.radiation.surface_heating
        + flux.radiation.space_heating,
        0.0,
        atol=2e-12,
    )
    # Existing temperature targets remain temperature targets, not encoded sunlight.
    np.testing.assert_allclose(initial.held_forcing.equilibrium_temperature, 260.0)


def test_donor_enthalpy_and_mechanical_work_close_instantaneous_global_energy(space):
    model = model_at(space, boundary())
    initial = model.initialize(
        temperature=280.0,
        vapor=0.002,
        liquid=0.001,
        ice=0.0001,
        surface_temperature=295.0,
        east=0.0,
    )
    rate, process, _, _ = model.tendency(initial.state, initial.held_forcing)
    _, power = jax.jvp(
        lambda state: model.inventories(state)[2], (initial.state,), (rate,)
    )
    area = 4 * np.pi * space.radius**2
    assert bool(process.successful)
    np.testing.assert_allclose(power / area, 0.0, atol=2e-8)
    np.testing.assert_allclose(process.energy_power / area, 0.0, atol=2e-8)


def test_slab_capacity_changes_integrated_sst_response(space):
    models = (
        model_at(space, boundary(capacity=2e6)),
        model_at(space, boundary(capacity=2e7)),
    )
    changes = []
    for model in models:
        initial = initialize(model)
        result = eqx.filter_jit(model.advance)(initial)
        assert bool(result.evidence.accepted)
        state = result.continuation.state
        temperature = model.plan.processes.surface_physics.temperature(
            state.surface_water, state.surface_energy, model.plan.processes.thermodynamics
        )
        changes.append(float(jnp.mean(temperature) - 290.0))
    assert changes[0] * changes[1] > 0
    assert abs(changes[0]) > 2 * abs(changes[1])


def test_surface_depletion_rejects_every_coupled_inventory_atomically(space):
    model = model_at(space, boundary())
    initial = initialize(model, water=1e-12)
    result = eqx.filter_jit(model.advance)(initial)
    assert not bool(result.evidence.accepted)
    assert not bool(result.evidence.admissible)
    expected = eqx.tree_at(
        lambda c: c.rejected_steps, initial, initial.rejected_steps + 1
    )
    for got, want in zip(
        jax.tree_util.tree_leaves(result.continuation),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        np.testing.assert_array_equal(got, want)


def test_native_restart_is_bitwise_and_rejects_changed_numeric_physics(space, tmp_path):
    model = model_at(space, boundary())
    initial = initialize(model)
    advance = eqx.filter_jit(model.advance)
    first = advance(initial)
    assert bool(first.evidence.accepted)
    path = tmp_path / "surface.npz"
    write_global_atmosphere_checkpoint(path, model, first.continuation)
    restored = read_global_atmosphere_checkpoint(path, model, initial)
    direct, restart = advance(first.continuation), advance(restored)
    assert bool(direct.evidence.accepted) and bool(restart.evidence.accepted)
    for got, want in zip(
        jax.tree_util.tree_leaves(direct), jax.tree_util.tree_leaves(restart), strict=True
    ):
        np.testing.assert_array_equal(got, want)
    for selector in (
        lambda m: m.plan.processes.surface_physics.radiation.longwave_absorption_scale,
        lambda m: m.plan.processes.surface_physics.slab.dry_heat_capacity,
    ):
        changed = eqx.tree_at(selector, model, replace_fn=lambda value: value * 1.1)
        with pytest.raises(ValueError):
            read_global_atmosphere_checkpoint(path, changed, initial)


def test_interactive_boundary_cannot_double_own_prescribed_fluxes():
    surface = boundary()
    for overlap in (
        {"evaporation_flux": 1e-5},
        {"sensible_heat_flux": 1.0},
        {"radiative_timescale": 86400.0},
        {"held_suarez": True},
    ):
        with pytest.raises(ValueError):
            GlobalAtmosphereProcesses(
                thermodynamics=surface.slab.thermodynamics,
                surface_physics=surface,
                **overlap,
            )


def test_global_flux_preconditioning_reduces_actual_boundary_residuals(space):
    model = model_at(space, boundary(capacity=2e7, p2=-0.48))
    initial = initialize(model, temperature=290.0)
    initial_residual, initial_valid = global_flux_residuals(model, initial)
    result = precondition_global_fluxes(model, initial)
    final_residual, final_valid = global_flux_residuals(model, result.continuation)
    assert bool(initial_valid & final_valid & result.successful)
    assert float(jnp.max(jnp.abs(initial_residual))) > 10.0
    assert float(jnp.max(jnp.abs(final_residual))) < 0.1
    assert float(jnp.max(jnp.abs(final_residual))) < (
        1e-3 * float(jnp.max(jnp.abs(initial_residual)))
    )
    assert abs(float(result.air_temperature_offset)) < 40
    assert abs(float(result.surface_temperature_offset)) < 40
    np.testing.assert_allclose(
        result.preparation_energy_j,
        model.inventories(result.continuation.state)[2]
        - model.inventories(initial.state)[2],
        rtol=2e-13,
    )
    rate, _, _, _ = model.tendency(
        result.continuation.state, result.continuation.held_forcing
    )
    assert float(jnp.max(jnp.abs(model.reconstruct(rate.temperature)))) > 0


def test_failed_global_flux_preconditioning_is_atomic(space):
    model = model_at(space, boundary(capacity=2e7, p2=-0.48))
    initial = initialize(model, temperature=290.0)
    result = precondition_global_fluxes(
        model,
        initial,
        maximum_surface_temperature_shift=0.01,
    )
    assert not bool(result.successful)
    for actual, expected in zip(
        jax.tree.leaves(result.continuation),
        jax.tree.leaves(initial),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(result.preparation_energy_j, 0.0)
    np.testing.assert_array_equal(
        result.final_flux_residual_w_per_m2,
        result.initial_flux_residual_w_per_m2,
    )


def test_interventions_share_baseline_preconditioned_physical_initial_state():
    baseline_model, baseline, baseline_evidence = make_model(
        scenario="baseline",
        bandlimit=3,
        levels=2,
        dt=20.0,
        initialization="flux-preconditioned",
    )
    forced_model, forced, forced_evidence = make_model(
        scenario="greenhouse",
        bandlimit=3,
        levels=2,
        dt=20.0,
        initialization="flux-preconditioned",
    )
    assert baseline_evidence["successful"]
    assert forced_evidence["successful"]
    assert forced_evidence["method"] == "common-baseline-flux-preconditioned-state"
    baseline_view = baseline_model.view(baseline.state)
    forced_view = forced_model.view(forced.state)
    for baseline_value, forced_value in (
        (baseline_view.temperature, forced_view.temperature),
        (baseline_view.surface_pressure, forced_view.surface_pressure),
        (baseline_view.east, forced_view.east),
        (baseline_view.north, forced_view.north),
        *zip(baseline_view.water, forced_view.water, strict=True),
    ):
        np.testing.assert_allclose(forced_value, baseline_value, rtol=2e-13)
    baseline_sst = baseline_model.plan.processes.surface_physics.temperature(
        baseline.state.surface_water,
        baseline.state.surface_energy,
        baseline_model.plan.processes.thermodynamics,
    )
    forced_sst = forced_model.plan.processes.surface_physics.temperature(
        forced.state.surface_water,
        forced.state.surface_energy,
        forced_model.plan.processes.thermodynamics,
    )
    np.testing.assert_allclose(forced_sst, baseline_sst, rtol=2e-13)
