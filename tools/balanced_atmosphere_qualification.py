# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Measured balance, wave, transport and integrated-AD qualification, not climate skill.

JAX_ENABLE_X64=1 python tools/balanced_atmosphere_qualification.py --output result.json
The default short campaign is intentionally small. Criteria are emitted with the
measurements; an unmet criterion or rejected step exits unsuccessfully. No paper
reference trajectory or high-resolution weather result is manufactured here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.atmosphere._balanced import DryGradientWindReference
from phydrax.applications.atmosphere._global import GlobalPrimitiveEquationPlan
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.atmosphere._processes import GlobalAtmosphereProcesses
from phydrax.applications.geophysics._vertical import HybridPressureCoordinate
from phydrax.discretization.spectral._spherical import SphericalSpectralPlan


SOURCE = "https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/pjp/qj_jablonowski_williamson_2006.pdf"


def model_at(bandlimit, levels, dt, **kwargs):
    space = SphericalSpectralPlan(bandlimit, sampling="gl").prepare(radius=6.371e6)
    sigma = np.linspace(0.0, 1.0, levels + 1)
    vertical = HybridPressureCoordinate(0.1 * (1 - sigma), sigma)
    return GlobalPrimitiveEquationPlan(space, vertical, dt=dt, **kwargs).prepare()


def area(model):
    return 4 * jnp.pi * model.plan.space.radius**2


def physical_mode(model):
    theta = model.work_space.transform.theta[:, None, None]
    longitude = model.work_space.transform.phi[None, :, None]
    return jnp.sin(theta) * jnp.cos(theta) * jnp.cos(longitude)


@eqx.filter_jit
def rollout(model, initial, steps):
    """Advance the real owner and trapezoid-integrate a physical enthalpy mode."""
    mode = physical_mode(model)

    def observable(state):
        view = model.view(state)
        return jnp.sum(
            model.work_space.integral(
                view.layer_mass * view.heat_capacity * view.temperature * mode
            )
        ) / area(model)

    def step(carry, _):
        current, integral = carry
        result = model.advance(current)
        integrated = integral + 0.5 * model.plan.dt * (
            observable(current.state) + observable(result.continuation.state)
        )
        return (result.continuation, integrated), result.evidence

    return jax.lax.scan(step, (initial, jnp.asarray(0.0)), xs=None, length=steps)


def summarize(model, initial, final, evidence):
    before, after = model.inventories(initial.state), model.inventories(final.state)
    duration = float(final.time - initial.time)
    angular_before, angular_after = (
        model.angular_momentum(initial.state),
        model.angular_momentum(final.state),
    )
    return {
        "all_accepted": bool(jnp.all(evidence.accepted)),
        "duration_seconds": duration,
        "energy_residual_w_per_m2": float(
            final.ledger.energy_residual / (area(model) * duration)
        )
        if duration > 0
        else None,
        "maximum_step_energy_residual_w_per_m2": float(
            jnp.max(jnp.abs(model.step_energy_flux(evidence)))
        ),
        "filter_work_joule": float(final.ledger.filter_energy),
        "atmosphere_plus_surface_mass_change_kg": float(after[0] - before[0]),
        "relative_water_inventory_change": float((after[1] - before[1]) / before[1]),
        "angular_momentum_change_kg_m2_per_s": float(angular_after - angular_before),
        "mean_angular_momentum_tendency_nm": float(
            (angular_after - angular_before) / duration
        )
        if duration > 0
        else None,
        "maximum_courant": float(jnp.max(evidence.advective_courant)),
    }


def identity_measurements(reference):
    # Centered differences do not call balance_residuals or its AD derivatives.
    p = np.geomspace(1200.0, 110000.0, 11)[:, None]
    lat = np.linspace(-np.pi / 2, np.pi / 2, 15)[None, :]
    inner_lat = lat[:, 1:-1]
    dx, dlat = 2e-4, 2e-4
    field = reference.fields(p, inner_lat)
    upper, lower = (
        reference.fields(p * np.exp(dx), inner_lat),
        reference.fields(p * np.exp(-dx), inner_lat),
    )
    north, south = (
        reference.fields(p, inner_lat + dlat),
        reference.fields(p, inner_lat - dlat),
    )
    hydro = (np.asarray(upper.geopotential) - np.asarray(lower.geopotential)) / (
        2 * dx
    ) + reference.gas_constant * np.asarray(field.temperature)
    grad = (np.asarray(north.geopotential) - np.asarray(south.geopotential)) / (
        2 * dlat * reference.radius
    )
    grad += 2 * reference.rotation_rate * np.sin(inner_lat) * np.asarray(field.east)
    grad += np.asarray(field.east) ** 2 * np.tan(inner_lat) / reference.radius
    # Independently difference du/dlog(p); no use of the parameter 'shear' here.
    du = (np.asarray(upper.east) - np.asarray(lower.east)) / (2 * dx)
    thermal_force = (
        2 * reference.rotation_rate * np.sin(inner_lat)
        + 2 * np.asarray(field.east) * np.tan(inner_lat) / reference.radius
    ) * du
    temperature_gradient = (
        reference.gas_constant
        / reference.radius
        * (np.asarray(north.temperature) - np.asarray(south.temperature))
        / (2 * dlat)
    )
    ad = reference.balance_residuals(p, lat)
    ps = reference.surface_pressure(lat)
    return {
        "reference_id": reference.reference_id,
        "nonzero_shear_m_per_s": reference.shear,
        "finite_difference_hydrostatic_max_j_per_kg": float(np.max(np.abs(hydro))),
        "finite_difference_gradient_wind_max_m_per_s2": float(np.max(np.abs(grad))),
        "finite_difference_thermal_wind_max_m_per_s2": float(
            np.max(np.abs(temperature_gradient - thermal_force))
        ),
        "thermal_wind_signal_m_per_s2": float(np.max(np.abs(thermal_force))),
        "ad_hydrostatic_max_j_per_kg": float(jnp.max(jnp.abs(ad[0]))),
        "ad_gradient_wind_max_m_per_s2": float(jnp.max(jnp.abs(ad[1]))),
        "ad_thermal_wind_max_m_per_s2": float(jnp.max(jnp.abs(ad[2]))),
        "surface_geopotential_max_m2_per_s2": float(
            jnp.max(jnp.abs(reference.fields(ps, lat).geopotential))
        ),
        "criteria": {
            "hydrostatic_fd_absolute_j_per_kg": 1e-5,
            "gradient_and_thermal_fd_absolute_m_per_s2": 1e-9,
            "nonzero_thermal_wind_signal_required": True,
        },
        "passed": bool(
            np.max(np.abs(hydro)) < 1e-5
            and np.max(np.abs(grad)) < 1e-9
            and np.max(np.abs(temperature_gradient - thermal_force)) < 1e-9
            and np.max(np.abs(thermal_force)) > 1e-5
        ),
    }


def resolution_measurements(reference, bandlimits, levels, dt, steps):
    horizontal, vertical = [], []
    for label, pairs, rows in (
        ("horizontal", [(l, max(levels)) for l in bandlimits], horizontal),
        ("vertical", [(max(bandlimits), n) for n in levels], vertical),
    ):
        for bandlimit, count in pairs:
            model = model_at(bandlimit, count, dt)
            initial = reference.initialize(model)
            diag = reference.diagnostics(model, initial)
            rates = model.budget_rates(initial.state, initial.held_forcing)
            (final, _), evidence = rollout(model, initial, steps)
            old, new = model.view(initial.state), model.view(final.state)
            row = {
                "sweep": label,
                "bandlimit": bandlimit,
                "levels": count,
                "surface_pressure_projection_pa": float(
                    diag.surface_pressure_projection_pa
                ),
                "temperature_projection_kelvin": float(
                    diag.temperature_projection_kelvin
                ),
                "wind_projection_m_per_s": float(diag.wind_projection_m_per_s),
                "finite_layer_geopotential_error_m2_per_s2": float(
                    diag.hydrostatic_geopotential_error_m2_per_s2
                ),
                "initial_acceleration_rms_m_per_s2": float(
                    diag.acceleration_rms_m_per_s2
                ),
                "initial_acceleration_max_m_per_s2": float(
                    diag.acceleration_max_m_per_s2
                ),
                "initial_temperature_tendency_kelvin_per_s": float(
                    diag.temperature_tendency_max_kelvin_per_s
                ),
                "initial_pressure_tendency_pa_per_s": float(
                    diag.surface_pressure_tendency_max_pa_per_s
                ),
                "spurious_wind_max_m_per_s": float(
                    jnp.max(jnp.hypot(new.east - old.east, new.north - old.north))
                ),
                "instantaneous_energy_residual_w_per_m2": float(
                    rates.energy_residual_w_per_m2
                ),
                "instantaneous_torque_residual_nm": float(rates.torque_residual_nm),
                **summarize(model, initial, final, evidence),
            }
            rows.append(row)
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "criteria": {
            "horizontal_pressure_projection_must_decrease": True,
            "vertical_initial_acceleration_must_decrease": True,
            "finest_initial_acceleration_max_m_per_s2": 1e-4,
            "maximum_energy_residual_w_per_m2": 1.0,
        },
        "passed": bool(
            all(
                r["all_accepted"] and r["maximum_step_energy_residual_w_per_m2"] < 1
                for r in horizontal + vertical
            )
            and horizontal[-1]["surface_pressure_projection_pa"]
            < horizontal[0]["surface_pressure_projection_pa"]
            and vertical[-1]["initial_acceleration_rms_m_per_s2"]
            < vertical[0]["initial_acceleration_rms_m_per_s2"]
            and vertical[-1]["initial_acceleration_max_m_per_s2"] < 1e-4
        ),
        "boundary": (
            "Horizontal RHS error can reach the fixed vertical-discretization "
            "floor. This is not exact discrete balance."
        ),
    }


def wave_measurements(dt, steps):
    """Independent one-layer oscillator, derived without the owner's fast matrix."""
    rows = []
    for refinement in (1, 2, 4):
        model = model_at(4, 1, dt / refinement, rotation_rate=0.0)
        initial = model.initialize(temperature=288.0 + 1e-3 * physical_mode(model))
        (final, _), evidence = rollout(model, initial, steps * refinement)
        # At rest, d_t=k²(h*T'+R*T0*ps'/ps0), T_t=-T0*h*d/cp,
        # ps_t=-delta_p*d; h=R*log(ps0/p_mid). This scalar derivation is
        # independent of construction, eigensystems and expm of fast_matrix.
        r = 0.1
        h = model.plan.gas_constant * np.log(2 / (1 + r))
        k2 = 6 / model.plan.space.radius**2
        frequency = np.sqrt(
            k2
            * (
                288.0 * h**2 / model.plan.heat_capacity
                + model.plan.gas_constant * 288.0 * (1 - r)
            )
        )
        center = model.plan.space.layout.bandlimit - 1

        def oscillator(state):
            c = (
                h * state.temperature[2, center + 1, 0]
                + model.plan.gas_constant
                * 288.0
                / 1e5
                * state.surface_pressure[2, center + 1]
            )
            return c + 1j * frequency / k2 * state.divergence[2, center + 1, 0]

        ratio = complex(oscillator(final.state) / oscillator(initial.state))
        phase = frequency * steps * dt
        rows.append(
            {
                "dt_seconds": dt / refinement,
                "frequency_rad_per_s": float(frequency),
                "expected_phase_rad": float(phase),
                "measured_phase_rad": float(np.angle(ratio)),
                "phase_error_rad": float(np.angle(ratio * np.exp(-1j * phase))),
                "amplitude_ratio": float(abs(ratio)),
                "complex_oscillator_error": float(abs(ratio - np.exp(1j * phase))),
                **summarize(model, initial, final, evidence),
            }
        )
    return {
        "refinement": rows,
        "criteria": {
            "phase_error_rad": 1e-3,
            "amplitude_error": 1e-3,
            "time_error_must_decrease": True,
        },
        "passed": bool(
            all(r["all_accepted"] for r in rows)
            and abs(rows[-1]["phase_error_rad"]) < 1e-3
            and abs(rows[-1]["amplitude_ratio"] - 1) < 1e-3
            and rows[-1]["complex_oscillator_error"] < rows[0]["complex_oscillator_error"]
        ),
        "boundary": (
            "Independent exact one-layer semidiscrete hydrostatic gravity-wave "
            "phase and neutral growth; not a continuum baroclinic instability "
            "reference."
        ),
    }


def torque_measurements():
    """Nonzero external torque, not only a symmetric zero-source identity."""
    model = model_at(6, 2, 30.0, processes=GlobalAtmosphereProcesses(held_suarez=True))
    speed = 20.0
    initial = model.initialize(
        east=speed * jnp.sin(model.work_space.transform.theta)[:, None, None]
    )
    rates = model.budget_rates(initial.state, initial.held_forcing)
    mass = 4 * np.pi * model.plan.space.radius**2 * 90000 / model.plan.gravity
    # Integrate cos²(latitude) independently (spherical mean=2/3).
    expected_momentum = (
        mass
        * 2
        / 3
        * model.plan.space.radius
        * (speed + model.plan.rotation_rate * model.plan.space.radius)
    )
    # The two equal-mass layers have sigma_mid=.325,.775. Only the
    # lower layer feels Held--Suarez Rayleigh drag.
    drag = (0.775 - 0.7) / (0.3 * 86400)
    expected_torque = -mass / 2 * 2 / 3 * model.plan.space.radius * speed * drag
    relative_error = abs(
        float(rates.angular_momentum_tendency_nm) - expected_torque
    ) / abs(expected_torque)
    return {
        "atmospheric_angular_momentum_kg_m2_per_s": float(
            rates.angular_momentum_kg_m2_per_s
        ),
        "analytic_angular_momentum_kg_m2_per_s": expected_momentum,
        "actual_rhs_torque_nm": float(rates.angular_momentum_tendency_nm),
        "independent_analytic_drag_torque_nm": expected_torque,
        "process_torque_nm": float(rates.process_torque_nm),
        "terrain_torque_nm": float(rates.terrain_torque_nm),
        "torque_residual_nm": float(rates.torque_residual_nm),
        "relative_torque_error": relative_error,
        "energy_residual_w_per_m2": float(rates.energy_residual_w_per_m2),
        "criteria": {"relative_torque_error": 1e-10},
        "passed": relative_error < 1e-10,
        "boundary": (
            "External Rayleigh torque on atmospheric angular momentum. No "
            "velocity or moment of inertia is stored for the surface/environment "
            "reservoirs."
        ),
    }


def perturbation_and_ad(reference, bandlimit, levels, dt, steps):
    rows, action_values, trajectory_errors = [], [], []
    final_states = []
    amplitude = 0.1
    for refinement in (1, 2, 4):
        model = model_at(bandlimit, levels, dt / refinement)
        base = reference.initialize(model)
        delta = model.project(
            jnp.broadcast_to(
                physical_mode(model), model.work_space.sample_shape + (levels,)
            )
        )

        def initial_at(value):
            return eqx.tree_at(
                lambda c: c.state.temperature,
                base,
                base.state.temperature + value * delta,
            )

        def action(value):
            (final, integrated), evidence = rollout(
                model, initial_at(value), steps * refinement
            )
            return integrated, jnp.all(evidence.accepted)

        value, derivative, ad_accepted = jax.jvp(
            action, (jnp.asarray(amplitude),), (jnp.asarray(1.0),), has_aux=True
        )
        finite_differences = []
        for epsilon in (1e-3, 3e-4):
            high, high_accepted = action(amplitude + epsilon)
            low, low_accepted = action(amplitude - epsilon)
            estimate = (high - low) / (2 * epsilon)
            finite_differences.append(
                {
                    "epsilon_kelvin": epsilon,
                    "all_accepted": bool(high_accepted & low_accepted),
                    "finite_difference_action_j_s_per_m2_per_kelvin": float(estimate),
                    "relative_error": float(
                        jnp.abs(estimate - derivative) / jnp.abs(derivative)
                    ),
                }
            )
        initial = initial_at(amplitude)
        (final, _), evidence = rollout(model, initial, steps * refinement)
        (unperturbed, _), baseline_evidence = rollout(model, base, steps * refinement)
        view, background = model.view(final.state), model.view(unperturbed.state)
        center = bandlimit - 1
        ratio = complex(
            (
                final.state.temperature[2, center + 1, 0]
                - unperturbed.state.temperature[2, center + 1, 0]
            )
            / (amplitude * delta[2, center + 1, 0])
        )
        eddy_energy = (
            0.5
            * jnp.sum(
                model.work_space.integral(
                    view.layer_mass
                    * (
                        (view.east - background.east) ** 2
                        + (view.north - background.north) ** 2
                    )
                )
            )
            / area(model)
        )
        rows.append(
            {
                "dt_seconds": dt / refinement,
                "integrated_enthalpy_mode_j_s_per_m2": float(value),
                "ad_action_j_s_per_m2_per_kelvin": float(derivative),
                "ad_trajectory_all_accepted": bool(ad_accepted),
                "finite_difference": finite_differences,
                "perturbation_temperature_mode_phase_rad": float(np.angle(ratio)),
                "perturbation_temperature_mode_amplitude_ratio": float(abs(ratio)),
                "perturbation_kinetic_energy_j_per_m2": float(eddy_energy),
                "baseline_all_accepted": bool(jnp.all(baseline_evidence.accepted)),
                **summarize(model, initial, final, evidence),
            }
        )
        action_values.append(float(derivative))
        final_states.append(final.state)
    # Identical retained spaces, equal physical time, finest trajectory reference.
    for state in final_states[:-1]:
        view, exact = model.view(state), model.view(final_states[-1])
        trajectory_errors.append(
            float(
                jnp.sqrt(
                    jnp.sum(
                        model.work_space.integral(
                            exact.layer_mass
                            * (
                                (view.east - exact.east) ** 2
                                + (view.north - exact.north) ** 2
                            )
                        )
                    )
                    / jnp.sum(model.work_space.integral(exact.layer_mass))
                )
            )
        )
    return {
        "refinement": rows,
        "wind_rms_error_against_dt_quarter_m_per_s": trajectory_errors,
        "ad_action_coarse_to_fine_difference": abs(action_values[0] - action_values[2]),
        "ad_action_half_to_fine_difference": abs(action_values[1] - action_values[2]),
        "criteria": {
            "ad_fd_relative_error": 1e-5,
            "trajectory_and_action_time_error_must_decrease": True,
        },
        "passed": bool(
            all(
                r["all_accepted"]
                and r["baseline_all_accepted"]
                and r["ad_trajectory_all_accepted"]
                and all(
                    e["all_accepted"] and e["relative_error"] < 1e-5
                    for e in r["finite_difference"]
                )
                for r in rows
            )
            and trajectory_errors[1] < trajectory_errors[0]
            and abs(action_values[1] - action_values[2])
            < abs(action_values[0] - action_values[2])
        ),
        "boundary": (
            "Finite-amplitude sheared-reference perturbation versus its separately "
            "evolved baseline and temporal self-convergence only. No published "
            "growth rate exists for this family; AD is for the accepted finite-step "
            "map, not an unqualified climate sensitivity."
        ),
    }


def terrain_and_transport(dt, steps):
    # Terrain supports isothermal rest only. Also expose the actual stratified
    # resting residual without requiring an invariant the owner does not have.
    space = SphericalSpectralPlan(6, sampling="gl").prepare(radius=6.371e6)
    terrain = (
        100
        * jnp.sin(space.transform.theta)[:, None]
        * jnp.cos(space.transform.phi)[None, :]
    )
    vertical = HybridPressureCoordinate([0.1, 0.05, 0.0], [0.0, 0.5, 1.0])
    model = GlobalPrimitiveEquationPlan(space, vertical, terrain=terrain, dt=dt).prepare()
    rest = model.initialize()
    (final, _), evidence = rollout(model, rest, steps)
    rest_rate, _, _, _ = model.tendency(rest.state, rest.held_forcing)
    u, v = model.vectors.wind(
        model.lift(rest_rate.vorticity), model.lift(rest_rate.divergence)
    )
    # Analytic dry stratification T=T0*(p/p_ref)^k has
    # Phi=R*T0/k * [1-(p/p_ref)^k] and a flat isobaric reference.
    # Using its terrain ps and temperature isolates the owner's finite-layer
    # hydrostatic/metric error; this is not an arbitrarily unbalanced profile.
    k = 0.05
    ps = 1e5 * (1 - k * model.terrain / (287.05 * 288.0)) ** (1 / k)
    interfaces = model.plan.vertical.interfaces(ps)
    stratified = model.initialize(
        surface_pressure=ps,
        temperature=288.0
        * (0.5 * (interfaces[..., :-1] + interfaces[..., 1:]) / 1e5) ** k,
    )
    rate, _, _, _ = model.tendency(stratified.state, stratified.held_forcing)
    su, sv = model.vectors.wind(model.lift(rate.vorticity), model.lift(rate.divergence))
    terrain_rates = model.budget_rates(stratified.state, stratified.held_forcing)

    # Water is active, not a passive arbitrary tracer. Dilute unsaturated vapor
    # tends to the independently known solid-rotation transport solution, but
    # finite vapor feedback and phase closure remain present and are declared.
    thermo = MoistThermodynamicPlan()
    tracer_model = model_at(
        6,
        2,
        dt,
        processes=GlobalAtmosphereProcesses(thermodynamics=thermo),
        heat_capacity=thermo.dry_cv + thermo.dry_gas_constant,
    )
    reference = DryGradientWindReference(
        shear=0.0, heat_capacity=tracer_model.plan.heat_capacity
    )
    theta = tracer_model.work_space.transform.theta[:, None, None]
    longitude = tracer_model.work_space.transform.phi[None, :, None]
    tracer = 1e-9 * (1 + 0.2 * jnp.sin(theta) * jnp.cos(longitude))
    tracer_initial = tracer_model.initialize(
        surface_pressure=reference.surface_pressure(jnp.pi / 2 - theta[..., 0]),
        east=reference.speed * jnp.sin(theta),
        vapor=tracer,
        surface_water=0.0,
    )
    (tracer_final, _), tracer_evidence = rollout(tracer_model, tracer_initial, steps)
    tracer_view = tracer_model.view(tracer_final.state)
    exact = 1e-9 * (
        1
        + 0.2
        * jnp.sin(theta)
        * jnp.cos(longitude - reference.speed / reference.radius * dt * steps)
    )
    transport_error = float(jnp.max(jnp.abs(tracer_view.water[0] - exact)) / 1e-9)
    return {
        "isothermal_terrain": {
            "actual_acceleration_max_m_per_s2": float(jnp.max(jnp.hypot(u, v))),
            **summarize(model, rest, final, evidence),
        },
        "stratified_terrain": {
            "analytic_profile_pressure_exponent": k,
            "actual_acceleration_max_m_per_s2": float(jnp.max(jnp.hypot(su, sv))),
            "terrain_torque_nm": float(terrain_rates.terrain_torque_nm),
            "torque_residual_nm": float(terrain_rates.torque_residual_nm),
            "boundary": (
                "Analytic stratified rest sampled onto two native finite layers: "
                "measured, not asserted to be well balanced."
            ),
        },
        "dilute_vapor_transport": {
            "relative_sup_error_against_solid_rotation": transport_error,
            "minimum_vapor_fraction": float(jnp.min(tracer_view.water[0])),
            "reference_rotation_angle_rad": reference.speed
            / reference.radius
            * dt
            * steps,
            **summarize(tracer_model, tracer_initial, tracer_final, tracer_evidence),
            "boundary": (
                "Smooth strictly positive dilute vapor with active thermodynamics, "
                "over the reported rotation angle; not monotone sharp-front or "
                "full-revolution qualification."
            ),
        },
        "criteria": {
            "isothermal_acceleration_m_per_s2": 1e-6,
            "dilute_transport_relative_sup_error": 1e-4,
        },
        "passed": bool(
            jnp.all(evidence.accepted)
            and jnp.all(tracer_evidence.accepted)
            and jnp.max(jnp.hypot(u, v)) < 1e-6
            and transport_error < 1e-4
            and jnp.min(tracer_view.water[0]) >= 0
        ),
    }


def rejected_boundaries():
    # Catch only the declared rejection class; an unexpected exception remains
    # a failure of the campaign rather than a false successful rejection.
    def rejected(call):
        try:
            call()
        except ValueError as error:
            return {"rejected": True, "reason": str(error)}
        return {"rejected": False}

    reference = DryGradientWindReference()
    space = SphericalSpectralPlan(4, sampling="gl").prepare(radius=6.371e6)
    terrain = 8e4 * jnp.cos(space.transform.theta)[:, None]
    moist = model_at(
        4,
        2,
        30.0,
        processes=GlobalAtmosphereProcesses(thermodynamics=MoistThermodynamicPlan()),
    )
    x = (
        jnp.sin(moist.work_space.transform.theta)[:, None, None]
        * jnp.cos(moist.work_space.transform.phi)[None, :, None]
    )
    cases = {
        "unrepresentable_degree_two_temperature": rejected(
            lambda: reference.initialize(model_at(2, 2, 30.0))
        ),
        "invalid_analytic_temperature_or_stability": rejected(
            lambda: DryGradientWindReference(shear=200.0)
        ),
        "pressure_domain_not_containing_surface": rejected(
            lambda: DryGradientWindReference(maximum_pressure=90000.0)
        ),
        "top_outside_reference_domain": rejected(
            lambda: DryGradientWindReference(minimum_pressure=20000.0).initialize(
                model_at(4, 2, 30.0)
            )
        ),
        "forcing_is_not_steady_dry_balance": rejected(
            lambda: reference.initialize(
                model_at(
                    4, 2, 30.0, processes=GlobalAtmosphereProcesses(held_suarez=True)
                )
            )
        ),
        "filtering_is_not_steady_balance": rejected(
            lambda: reference.initialize(model_at(4, 2, 30.0, filter_rate=1e-6))
        ),
        "terrain_not_in_flat_ground_family": rejected(
            lambda: reference.initialize(model_at(4, 2, 30.0, terrain=1.0))
        ),
        "unresolved_terrain_rest": rejected(
            lambda: model_at(4, 2, 30.0, terrain=terrain, terrain_rest_tolerance=1e-12)
        ),
        "positive_unresolved_water_plume": rejected(
            lambda: moist.initialize(vapor=1e-3 * ((1 + x) / 2) ** 8)
        ),
    }
    cases["out_of_domain_field_is_unsuccessful"] = {
        "rejected": not bool(reference.fields(100.0, 0.0).successful)
    }
    cases["nonphysical_latitude_is_unsuccessful"] = {
        "rejected": not bool(reference.fields(1e5, 2.0).successful)
    }
    return {"cases": cases, "passed": all(case["rejected"] for case in cases.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bandlimits", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--dt", type=float, default=120.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--wave-dt", type=float, default=300.0)
    parser.add_argument("--wave-steps", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        len(args.bandlimits) < 2
        or len(args.levels) < 2
        or sorted(set(args.bandlimits)) != args.bandlimits
        or sorted(set(args.levels)) != args.levels
        or min(args.bandlimits) < 3
        or min(args.levels) < 1
        or not np.isfinite(args.dt)
        or not np.isfinite(args.wave_dt)
        or min(args.dt, args.wave_dt, args.steps, args.wave_steps) <= 0
    ):
        parser.error(
            "Require increasing resolution lists of length >=2, bandlimits >=3, and positive finite steps/dt."
        )
    reference = DryGradientWindReference()
    result = {
        "reference": {
            "family": "independently-derived-log-pressure-gradient-and-thermal-wind",
            "published_benchmark": False,
            "published_strategy_source": SOURCE,
            "source_boundary": (
                "Jablonowski and Williamson (2006), doi:10.1256/qj.06.12, "
                "motivates separate balance/evolution and resolution tests. Its "
                "jets, terrain, positive-top limit and high-resolution reference "
                "trajectories are NOT reproduced by this family."
            ),
            "parameters": {
                "speed_m_per_s": reference.speed,
                "shear_m_per_s": reference.shear,
                "temperature_kelvin": reference.temperature,
                "pressure_domain_pa": [
                    reference.minimum_pressure,
                    reference.maximum_pressure,
                ],
            },
        },
        "identities": identity_measurements(reference),
        "resolution": resolution_measurements(
            reference, args.bandlimits, args.levels, args.dt, args.steps
        ),
        "independent_gravity_wave": wave_measurements(args.wave_dt, args.wave_steps),
        "external_torque": torque_measurements(),
        "perturbation_and_integrated_ad": perturbation_and_ad(
            reference, max(args.bandlimits), max(args.levels), args.dt, args.steps
        ),
        "terrain_and_transport": terrain_and_transport(min(args.dt, 120.0), args.steps),
        "rejected_boundaries": rejected_boundaries(),
        "claim_boundary": (
            "Short-run numerical qualification only. No weather skill, validated "
            "baroclinic growth spectrum, monotone tracer transport, "
            "stratified-terrain well-balancing or equilibrated climate claim."
        ),
    }
    result["passed"] = all(
        result[key]["passed"]
        for key in (
            "identities",
            "resolution",
            "independent_gravity_wave",
            "external_torque",
            "perturbation_and_integrated_ad",
            "terrain_and_transport",
            "rejected_boundaries",
        )
    )
    text = json.dumps(result, indent=2, allow_nan=False)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
