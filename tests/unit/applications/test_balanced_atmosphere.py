# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.atmosphere._balanced import DryGradientWindReference
from phydrax.applications.atmosphere._global import GlobalPrimitiveEquationPlan
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.atmosphere._processes import GlobalAtmosphereProcesses
from phydrax.applications.geophysics._vertical import HybridPressureCoordinate
from phydrax.discretization.spectral._spherical import SphericalSpectralPlan


@pytest.fixture(scope="module")
def space():
    return SphericalSpectralPlan(6, sampling="gl").prepare(radius=6.371e6)


def model_at(space, levels=4, **kwargs):
    sigma = np.linspace(0, 1, levels + 1)
    return GlobalPrimitiveEquationPlan(
        space, HybridPressureCoordinate(0.1 * (1 - sigma), sigma), dt=30.0, **kwargs
    ).prepare()


def test_nonzero_shear_independently_satisfies_hydrostatic_and_thermal_wind():
    reference = DryGradientWindReference()
    pressure = np.geomspace(1200, 110000, 9)[:, None]
    latitude = np.linspace(-1.4, 1.4, 11)[None, :]
    epsilon = 1e-4
    field = reference.fields(pressure, latitude)
    top = reference.fields(pressure * np.exp(-epsilon), latitude)
    bottom = reference.fields(pressure * np.exp(epsilon), latitude)
    north = reference.fields(pressure, latitude + epsilon)
    south = reference.fields(pressure, latitude - epsilon)
    dphi_dlogp = (bottom.geopotential - top.geopotential) / (2 * epsilon)
    dphi_dlat = (north.geopotential - south.geopotential) / (2 * epsilon)
    dt_dlat = (north.temperature - south.temperature) / (2 * epsilon)
    du_dlogp = (bottom.east - top.east) / (2 * epsilon)
    np.testing.assert_allclose(
        dphi_dlogp, -reference.gas_constant * field.temperature, atol=2e-6, rtol=0
    )
    acceleration = (
        2 * reference.rotation_rate * np.sin(latitude) * field.east
        + field.east**2 * np.tan(latitude) / reference.radius
    )
    np.testing.assert_allclose(
        dphi_dlat / reference.radius, -acceleration, atol=1e-10, rtol=0
    )
    thermal_force = (
        2 * reference.rotation_rate * np.sin(latitude)
        + 2 * field.east * np.tan(latitude) / reference.radius
    ) * du_dlogp
    assert float(jnp.max(jnp.abs(thermal_force))) > 1e-5
    np.testing.assert_allclose(
        reference.gas_constant / reference.radius * dt_dlat,
        thermal_force,
        atol=1e-10,
        rtol=0,
    )


def test_surface_branch_and_zero_shear_equator_pole_limits():
    latitude = np.array([-np.pi / 2, -0.7, 0.0, 0.7, np.pi / 2])
    for shear in (-10.0, 0.0, 1e-12):
        reference = DryGradientWindReference(shear=shear)
        ps = reference.surface_pressure(latitude)
        np.testing.assert_allclose(
            reference.fields(ps, latitude).geopotential, 0.0, atol=3e-11
        )
        np.testing.assert_allclose(ps[2], reference.reference_pressure, atol=1e-10)
        hydro, gradient, thermal = reference.balance_residuals(30000.0, latitude)
        np.testing.assert_allclose(hydro, 0, atol=3e-11)
        np.testing.assert_allclose(gradient, 0, atol=3e-18)
        np.testing.assert_allclose(thermal, 0, atol=3e-18)
    reference = DryGradientWindReference(shear=0)
    expected = reference.reference_pressure * np.exp(
        -(
            reference.rotation_rate * reference.radius * reference.speed
            + reference.speed**2 / 2
        )
        * np.sin(latitude) ** 2
        / (reference.gas_constant * reference.temperature)
    )
    np.testing.assert_allclose(reference.surface_pressure(latitude), expected, rtol=3e-16)


def test_native_sheared_state_residual_decreases_with_vertical_resolution(space):
    reference = DryGradientWindReference()
    coarse, fine = model_at(space, 2), model_at(space, 8)
    initial_coarse, initial_fine = (
        reference.initialize(coarse),
        reference.initialize(fine),
    )
    coarse_error, fine_error = (
        reference.diagnostics(coarse, initial_coarse),
        reference.diagnostics(fine, initial_fine),
    )
    # This would fail a barotropic-only initializer or an unpaired pressure metric.
    assert float(fine_error.acceleration_rms_m_per_s2) < float(
        coarse_error.acceleration_rms_m_per_s2
    )
    view = fine.view(initial_fine.state)
    assert float(jnp.max(jnp.abs(view.east[..., 0] - view.east[..., -1]))) > 5
    result = fine.advance(initial_fine)
    assert bool(result.evidence.accepted)
    assert float(jnp.abs(fine.step_energy_flux(result.evidence))) < 1.0


def test_atmospheric_angular_momentum_and_external_drag_torque(space):
    model = model_at(space, 2, processes=GlobalAtmosphereProcesses(held_suarez=True))
    speed = 20.0
    initial = model.initialize(
        east=speed * jnp.sin(model.work_space.transform.theta)[:, None, None]
    )
    radius, omega = model.plan.space.radius, model.plan.rotation_rate
    atmospheric_mass = 4 * np.pi * radius**2 * 90000 / model.plan.gravity
    expected = atmospheric_mass * 2 / 3 * radius * (speed + omega * radius)
    np.testing.assert_allclose(
        model.angular_momentum(initial.state), expected, rtol=2e-13
    )
    rates = model.budget_rates(initial.state, initial.held_forcing)
    # Two equal pressure-thickness layers, bottom midpoint sigma=0.775:
    # Held--Suarez drag=(0.775-0.7)/(0.3*86400), only in that layer.
    drag = (0.775 - 0.7) / (0.3 * 86400)
    expected_torque = -atmospheric_mass / 2 * 2 / 3 * radius * speed * drag
    np.testing.assert_allclose(rates.process_torque_nm, expected_torque, rtol=2e-12)
    np.testing.assert_allclose(
        rates.angular_momentum_tendency_nm, expected_torque, rtol=2e-12
    )
    assert abs(float(rates.torque_residual_nm)) < 1e-11 * abs(expected_torque)


def test_energy_neutral_projection_closes_resolved_angular_momentum(space):
    model = model_at(
        space,
        2,
        processes=GlobalAtmosphereProcesses(held_suarez=True),
        angular_momentum_projection="energy-neutral",
        maximum_angular_momentum_projection_fraction=1e-2,
    )
    theta = model.work_space.transform.theta[:, None, None]
    phi = model.work_space.transform.phi[None, :, None]
    initial = model.initialize(
        east=15.0 * jnp.sin(theta) * (1.0 + 0.25 * jnp.sin(theta) * jnp.cos(phi)),
        north=3.0 * jnp.sin(theta) * jnp.sin(phi),
        temperature=280.0
        + 5.0 * jnp.sin(theta) ** 2
        + 0.5 * jnp.sin(theta) * jnp.cos(phi),
    )
    rates = model.budget_rates(initial.state, initial.held_forcing)
    torque_scale = max(
        abs(float(rates.angular_momentum_kg_m2_per_s)) / 86400.0,
        abs(float(rates.terrain_torque_nm)) + abs(float(rates.process_torque_nm)),
        1.0,
    )
    assert abs(float(rates.raw_torque_residual_nm)) > 0
    np.testing.assert_allclose(
        rates.raw_torque_residual_nm + rates.projection_torque_nm,
        0.0,
        atol=2e-12 * torque_scale,
    )
    assert abs(float(rates.torque_residual_nm)) < 2e-12 * torque_scale
    assert (
        float(rates.projection_fraction)
        < model.plan.maximum_angular_momentum_projection_fraction
    )
    assert abs(float(rates.projection_energy_power_w_per_m2)) < 1e-7
    result = model.advance(initial)
    assert bool(result.evidence.accepted)
    assert bool(result.evidence.angular_momentum_projection_successful)
    assert bool(result.evidence.angular_momentum_projection_evaluated)


def test_filter_angular_momentum_matches_independent_unfiltered_trajectory(space):
    plain = model_at(space, 2)
    filtered = model_at(space, 2, filter_rate=0.1)
    wind = 20.0 * jnp.sin(plain.work_space.transform.theta)[:, None, None]
    no_filter = plain.advance(plain.initialize(east=wind))
    with_filter = filtered.advance(filtered.initialize(east=wind))
    assert bool(no_filter.evidence.accepted)
    assert bool(with_filter.evidence.accepted)
    angular_change = filtered.angular_momentum(
        with_filter.continuation.state
    ) - plain.angular_momentum(no_filter.continuation.state)
    assert float(angular_change) < 0
    np.testing.assert_allclose(
        with_filter.evidence.filter_angular_momentum, angular_change, rtol=2e-13
    )
    np.testing.assert_array_equal(no_filter.evidence.filter_angular_momentum, 0.0)


def test_rejects_reference_domain_mismatch_and_unrepresentable_water(space):
    reference = DryGradientWindReference()
    with pytest.raises(ValueError):
        DryGradientWindReference(shear=200)
    with pytest.raises(ValueError):
        DryGradientWindReference(maximum_pressure=90000)
    assert not bool(reference.fields(100, 0).successful)
    assert not bool(reference.fields(100000, 2).successful)
    with pytest.raises(ValueError):
        reference.initialize(model_at(space, terrain=1.0))
    with pytest.raises(ValueError):
        DryGradientWindReference(minimum_pressure=20000).initialize(model_at(space))
    with pytest.raises(ValueError):
        reference.initialize(
            model_at(space, processes=GlobalAtmosphereProcesses(held_suarez=True))
        )
    low_space = SphericalSpectralPlan(2, sampling="gl").prepare(radius=6.371e6)
    with pytest.raises(ValueError):
        reference.initialize(model_at(low_space))
    # Spectral inventory transport does not promise positivity for unresolved
    # sharp profiles. Positive input samples can have a negative modal projection.
    sharp_space = SphericalSpectralPlan(4, sampling="gl").prepare(radius=6.371e6)
    moist = model_at(
        sharp_space,
        processes=GlobalAtmosphereProcesses(thermodynamics=MoistThermodynamicPlan()),
    )
    x = (
        jnp.sin(moist.work_space.transform.theta)[:, None, None]
        * jnp.cos(moist.work_space.transform.phi)[None, :, None]
    )
    with pytest.raises(ValueError):
        moist.initialize(vapor=1e-3 * ((1 + x) / 2) ** 8)
