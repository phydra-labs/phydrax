# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.atmosphere._global import (
    GlobalPrimitiveEquationPlan,
    read_global_atmosphere_checkpoint,
    write_global_atmosphere_checkpoint,
)
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.atmosphere._processes import GlobalAtmosphereProcesses
from phydrax.applications.geophysics._vertical import HybridPressureCoordinate
from phydrax.discretization.spectral._spherical import SphericalSpectralPlan


def _reconstruct_water_phases(model, water):
    coefficients = jnp.stack(water, axis=-1)
    flattened = coefficients.reshape(coefficients.shape[:2] + (-1,))
    return model.reconstruct(flattened).reshape(
        model.work_space.sample_shape + (model.levels, 3)
    )


@pytest.fixture(scope="module")
def space():
    return SphericalSpectralPlan(4, sampling="gl").prepare(radius=6.371e6)


def _model(space, **kwargs):
    vertical = HybridPressureCoordinate([0.1, 0.05, 0.0], [0.0, 0.5, 1.0])
    return GlobalPrimitiveEquationPlan(space, vertical, dt=20.0, **kwargs).prepare()


def test_isothermal_rest_and_closed_column_continuity(space):
    model = _model(space)
    initial = model.initialize()
    view = model.view(initial.state)
    psdot, flux, omega = model.continuity(view, jnp.zeros_like(view.layer_mass))
    np.testing.assert_allclose(flux[..., (0, -1)], 0.0, atol=1e-12)
    np.testing.assert_allclose(psdot, 0.0, atol=1e-12)
    np.testing.assert_allclose(omega, 0.0, atol=1e-12)
    result = model.advance(initial)
    assert bool(result.evidence.accepted)
    assert not bool(result.evidence.angular_momentum_projection_evaluated)
    final = model.view(result.continuation.state)
    np.testing.assert_allclose(final.east, 0.0, atol=1e-8)
    np.testing.assert_allclose(final.north, 0.0, atol=1e-8)
    np.testing.assert_allclose(final.temperature, view.temperature, atol=1e-9)
    np.testing.assert_allclose(final.surface_pressure, view.surface_pressure, atol=1e-7)


def test_fast_gravity_operator_is_actual_primitive_equation_linearization(space):
    model = _model(space, rotation_rate=0.0)
    initial = model.initialize()
    theta = model.work_space.transform.theta[:, None, None]
    phi = model.work_space.transform.phi[None, :, None]
    harmonic = jnp.sin(theta) * jnp.cos(phi)
    zero = jax.tree_util.tree_map(jnp.zeros_like, initial.state)
    perturbation = eqx.tree_at(
        lambda s: (s.divergence, s.temperature, s.surface_pressure),
        zero,
        (
            model.project(harmonic * jnp.asarray([1e-7, -3e-8])),
            model.project(harmonic * jnp.asarray([0.3, -0.2])),
            model.project((100 * harmonic)[..., 0]),
        ),
    )
    exact_linear = jax.jvp(
        lambda s: model.tendency(s, initial.held_forcing)[0],
        (initial.state,),
        (perturbation,),
    )[1]
    fast = model.fast_tendency(perturbation)
    np.testing.assert_allclose(
        exact_linear.divergence, fast.divergence, rtol=2e-7, atol=2e-16
    )
    np.testing.assert_allclose(
        exact_linear.temperature, fast.temperature, rtol=2e-7, atol=2e-11
    )
    np.testing.assert_allclose(
        exact_linear.surface_pressure, fast.surface_pressure, rtol=2e-7, atol=2e-10
    )
    # A lower-layer thermal perturbation changes upper-level acceleration:
    # hydrostatic vertical coupling is not a stack of independent wave columns.
    lower = eqx.tree_at(
        lambda s: s.temperature, zero, model.project(harmonic * jnp.asarray([0.0, 1.0]))
    )
    assert float(jnp.max(jnp.abs(model.fast_tendency(lower).divergence[..., 0]))) > 1e-12


def test_nonuniform_flow_conserves_pressure_mass_and_diagnoses_vertical_flux(space):
    model = _model(space, rotation_rate=0.0)
    theta = model.work_space.transform.theta[:, None, None]
    phi = model.work_space.transform.phi[None, :, None]
    initial = model.initialize(
        east=5 * jnp.sin(phi) * jnp.asarray([1.0, -0.3]),
        north=2 * jnp.sin(theta) * jnp.cos(phi),
        vapor=0.0,
    )
    view = model.view(initial.state)
    psdot, flux, _ = model.continuity(view, jnp.zeros_like(view.layer_mass))
    horizontal = model.reconstruct(
        model.divergence(view.layer_mass * view.east, view.layer_mass * view.north)
    )
    closure = (
        horizontal
        + jnp.diff(flux, axis=-1) / model.plan.gravity
        + jnp.diff(model.plan.vertical.b) * psdot[..., None] / model.plan.gravity
    )
    np.testing.assert_allclose(closure, 0.0, atol=2e-13)
    np.testing.assert_allclose(flux[..., (0, -1)], 0.0, atol=2e-12)
    assert float(jnp.max(jnp.abs(flux[..., 1:-1]))) > 1e-3
    result = model.advance(initial)
    assert bool(result.evidence.accepted)
    before, after = (
        model.inventories(initial.state)[0],
        model.inventories(result.continuation.state)[0],
    )
    np.testing.assert_allclose(after, before, rtol=2e-13)


def test_joint_water_projection_preserves_total_water_and_energy(space):
    thermodynamics = MoistThermodynamicPlan()
    processes = GlobalAtmosphereProcesses(thermodynamics=thermodynamics)
    model = _model(
        space,
        processes=processes,
        water_limiter="conservative",
        maximum_water_phase_repartition_fraction=1e-2,
    )
    initial = model.initialize(temperature=280.0, vapor=0.005, liquid=1e-6)
    (
        unchanged,
        active,
        total_moved,
        total_relative,
        phase_moved,
        phase_relative,
        energy_residual,
        successful,
    ) = model.limit_water_inventories(initial.state)
    assert bool(successful)
    assert not bool(active)
    for value in (
        total_moved,
        total_relative,
        phase_moved,
        phase_relative,
        energy_residual,
    ):
        np.testing.assert_array_equal(value, 0.0)
    for actual, expected in zip(
        jax.tree.leaves(unchanged),
        jax.tree.leaves(initial.state),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)

    view = model.view(initial.state)
    theta = model.work_space.transform.theta[:, None, None]
    phi = model.work_space.transform.phi[None, :, None]
    signed_liquid = view.layer_mass * 1e-6 * (1.0 + 2.0 * jnp.sin(theta) * jnp.cos(phi))
    candidate = eqx.tree_at(
        lambda state: state.water,
        initial.state,
        (
            initial.state.water[0],
            model.project(signed_liquid),
            initial.state.water[2],
        ),
    )
    before_energy = model.inventories(candidate)[2]
    before_total = sum(candidate.water)
    (
        limited,
        active,
        total_moved,
        total_relative,
        phase_moved,
        phase_relative,
        energy_residual,
        successful,
    ) = model.limit_water_inventories(candidate)
    assert bool(active)
    assert bool(successful)
    assert bool(model.admissible(limited))
    assert float(phase_moved) > 0
    assert 0 < float(phase_relative) < model.plan.maximum_water_phase_repartition_fraction
    assert float(total_relative) < model.plan.water_projection_tolerance
    after_total = sum(limited.water)
    np.testing.assert_allclose(after_total, before_total, rtol=2e-13, atol=1e-12)
    np.testing.assert_allclose(
        model.reconstruct(after_total),
        model.reconstruct(before_total),
        rtol=2e-13,
        atol=1e-12,
    )

    area = 4 * jnp.pi * model.plan.space.radius**2
    original = _reconstruct_water_phases(model, candidate.water)
    constrained = _reconstruct_water_phases(model, limited.water)
    means = model.work_space.integral(original) / area
    minima = jnp.min(original, axis=(0, 1))
    factors = jnp.where(
        minima < 0,
        means / (means - minima),
        1.0,
    )
    contracted = means + factors * (original - means)
    contracted = contracted.at[..., 0].add(
        jnp.sum(original, axis=-1) - jnp.sum(contracted, axis=-1)
    )
    constrained_difference = constrained - original
    contraction_difference = contracted - original
    constrained_l2 = jnp.sqrt(
        jnp.sum(model.work_space.integral(constrained_difference**2))
    )
    contraction_l2 = jnp.sqrt(
        jnp.sum(model.work_space.integral(contraction_difference**2))
    )
    contraction_moved = 0.5 * jnp.sum(
        model.work_space.integral(jnp.abs(contraction_difference))
    )
    assert float(constrained_l2) < 0.8 * float(contraction_l2)
    assert float(phase_moved) < 0.8 * float(contraction_moved)
    np.testing.assert_allclose(model.inventories(limited)[2], before_energy, rtol=2e-13)
    assert float(jnp.abs(energy_residual) / area) < 1e-5


def test_joint_water_projection_rejects_excessive_phase_repartition(space):
    thermodynamics = MoistThermodynamicPlan()
    processes = GlobalAtmosphereProcesses(thermodynamics=thermodynamics)
    model = _model(
        space,
        processes=processes,
        water_limiter="conservative",
        maximum_water_phase_repartition_fraction=1e-12,
    )
    initial = model.initialize(temperature=280.0, vapor=0.005, liquid=1e-6)
    view = model.view(initial.state)
    theta = model.work_space.transform.theta[:, None, None]
    phi = model.work_space.transform.phi[None, :, None]
    candidate = eqx.tree_at(
        lambda state: state.water,
        initial.state,
        (
            initial.state.water[0],
            model.project(
                view.layer_mass * 1e-6 * (1.0 + 2.0 * jnp.sin(theta) * jnp.cos(phi))
            ),
            initial.state.water[2],
        ),
    )
    _, active, _, _, _, phase_relative, _, successful = model.limit_water_inventories(
        candidate
    )
    assert bool(active)
    assert float(phase_relative) > model.plan.maximum_water_phase_repartition_fraction
    assert not bool(successful)


def test_failed_step_preserves_physical_state_time_and_forcing(space):
    model = _model(space)
    initial = model.initialize()
    invalid = eqx.tree_at(
        lambda c: c.state.temperature, initial, -initial.state.temperature
    )
    result = model.advance(invalid)
    assert not bool(result.evidence.accepted)
    assert int(result.continuation.rejected_steps) == 1
    for before, after in zip(
        jax.tree_util.tree_leaves(invalid.state),
        jax.tree_util.tree_leaves(result.continuation.state),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(result.continuation.time, invalid.time)
    np.testing.assert_array_equal(result.continuation.forcing_age, invalid.forcing_age)


def test_moist_precipitation_budget_and_cadenced_restart(space, tmp_path):
    processes = GlobalAtmosphereProcesses(
        thermodynamics=MoistThermodynamicPlan(),
        cadence=3,
        condensation_timescale=300.0,
        precipitation_timescale=600.0,
        radiative_timescale=40 * 86400.0,
        sensible_heat_flux=5.0,
    )
    model = _model(space, processes=processes)
    initial = model.initialize(
        temperature=280.0, vapor=0.007, liquid=0.002, surface_water=20.0
    )
    first = model.advance(initial)
    assert bool(first.evidence.accepted)
    assert float(first.evidence.physical_phase_conversion_mass) > 0
    np.testing.assert_allclose(
        first.continuation.ledger.physical_phase_conversion_mass,
        first.evidence.physical_phase_conversion_mass,
    )
    np.testing.assert_array_equal(
        first.continuation.ledger.water_total_redistribution_mass, 0.0
    )
    np.testing.assert_array_equal(
        first.continuation.ledger.water_phase_repartition_mass, 0.0
    )
    assert (
        float(
            jnp.min(first.continuation.state.surface_water - initial.state.surface_water)
        )
        > 0
    )
    np.testing.assert_allclose(
        model.inventories(first.continuation.state)[1],
        model.inventories(initial.state)[1],
        rtol=2e-12,
    )
    # Independent closed atmosphere+surface+environment+upper-lid energy:
    # no process-JVP subtraction is allowed to manufacture this balance.
    np.testing.assert_allclose(
        model.inventories(first.continuation.state)[2],
        model.inventories(initial.state)[2],
        rtol=2e-7,
    )
    assert (
        float(jnp.max(jnp.abs(model.view(first.continuation.state).temperature - 280.0)))
        > 1e-4
    )
    path = write_global_atmosphere_checkpoint(
        tmp_path / "global.npz", model, first.continuation
    )
    restored = read_global_atmosphere_checkpoint(path, model, initial)
    uninterrupted, restarted = first.continuation, restored
    # Cross the held-forcing refresh boundary after loading, not only a static read.
    for _ in range(3):
        continuous_result, restarted_result = (
            model.advance(uninterrupted),
            model.advance(restarted),
        )
        assert bool(continuous_result.evidence.accepted) and bool(
            restarted_result.evidence.accepted
        )
        uninterrupted, restarted = (
            continuous_result.continuation,
            restarted_result.continuation,
        )
    for left, right in zip(
        jax.tree_util.tree_leaves(uninterrupted),
        jax.tree_util.tree_leaves(restarted),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    assert float(jnp.abs(uninterrupted.ledger.water_residual)) < 2e-12 * float(
        model.inventories(initial.state)[1]
    )


def test_unresolved_terrain_fails_rest_admission(space):
    terrain = 8e4 * jnp.cos(space.transform.theta)[:, None]
    with pytest.raises(ValueError, match="rest gate"):
        _model(space, terrain=terrain, terrain_rest_tolerance=1e-12)


def test_monopole_rest_has_no_diffusive_mode():
    space = SphericalSpectralPlan(1, sampling="gl").prepare(radius=6.371e6)
    model = _model(space, filter_rate=1.0)
    initial = model.initialize()
    result = model.advance(initial)
    assert bool(result.evidence.accepted)
    np.testing.assert_allclose(
        model.view(result.continuation.state).temperature, 288.0, atol=1e-10
    )
    np.testing.assert_allclose(result.evidence.filter_energy, 0.0, atol=1e-6)
