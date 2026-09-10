# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Physical initial-condition preconditioning for interactive global atmospheres."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import phydrax.linalg as la

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ._global import GlobalAtmosphereContinuation, PreparedGlobalAtmosphere


class GlobalFluxPreconditioningResult(StrictModule):
    """Atomic two-control preparation evidence, never an equilibrium claim."""

    continuation: GlobalAtmosphereContinuation
    air_temperature_offset: Array
    surface_temperature_offset: Array
    initial_flux_residual_w_per_m2: Array
    final_flux_residual_w_per_m2: Array
    residual_history_w_per_m2: Array
    accepted_iterations: Array
    jacobian_condition: Array
    preparation_energy_j: Array
    successful: Array
    preparation_id: str = eqx.field(static=True)


def global_flux_residuals(
    model: PreparedGlobalAtmosphere,
    continuation: GlobalAtmosphereContinuation,
    /,
) -> tuple[Array, Array]:
    """Return global-mean TOA/surface imbalance and physical validity."""
    if not isinstance(model, PreparedGlobalAtmosphere):
        raise TypeError("model must be a PreparedGlobalAtmosphere.")
    if not isinstance(continuation, GlobalAtmosphereContinuation):
        raise TypeError("continuation must be a GlobalAtmosphereContinuation.")
    processes = model.plan.processes
    boundary = processes.surface_physics
    if boundary is None or processes.thermodynamics is None:
        raise ValueError("Flux diagnostics require the interactive moist surface owner.")
    state = continuation.state
    view = model.view(state)
    flux = boundary.evaluate(
        processes.thermodynamics,
        view,
        state.surface_water,
        state.surface_energy,
        continuation.held_forcing.solar_down,
    )
    mass = view.layer_mass
    mechanical = view.geopotential + 0.5 * (view.east**2 + view.north**2)
    _, _, liquid_enthalpy, ice_enthalpy = processes.thermodynamics.phase_enthalpies(
        view.temperature
    )
    rain_energy = (
        jnp.sum(
            mass
            * (
                view.water[1] * (liquid_enthalpy + mechanical)
                + view.water[2] * (ice_enthalpy + mechanical)
            ),
            axis=-1,
        )
        / processes.precipitation_timescale
    )
    surface_power = (
        flux.radiation.surface_heating
        - flux.exchange.sensible_heat
        - flux.exchange.water_enthalpy
        - flux.exchange.water_mass * mechanical[..., -1]
        + rain_energy
    )
    area = 4 * jnp.pi * model.plan.space.radius**2
    toa = -model.work_space.integral(flux.radiation.space_heating) / area
    surface = model.work_space.integral(surface_power) / area
    residual = jnp.asarray((toa, surface))
    valid = (
        model.admissible(state)
        & jnp.all(flux.successful)
        & jnp.all(jnp.isfinite(residual))
    )
    return residual, valid


def precondition_global_fluxes(
    model: PreparedGlobalAtmosphere,
    continuation: GlobalAtmosphereContinuation,
    /,
    *,
    maximum_steps: int = 8,
    flux_tolerance_w_per_m2: float = 0.1,
    maximum_air_temperature_shift: float = 40.0,
    maximum_surface_temperature_shift: float = 40.0,
    maximum_jacobian_condition: float = 1.0e8,
) -> GlobalFluxPreconditioningResult:
    """Reduce global-mean TOA and surface fluxes with two bounded temperature offsets.

    The atmospheric lapse-rate and every resolved anomaly are retained; one uniform
    air-temperature offset and one uniform wet-slab-temperature offset are solved.
    This is an instantaneous flux preconditioner. It does not make local layer
    heating vanish, balance horizontal pressure gradients, or certify stationarity.
    Failed preparation is atomic and returns the input continuation.
    """
    if not isinstance(model, PreparedGlobalAtmosphere):
        raise TypeError("model must be a PreparedGlobalAtmosphere.")
    if not isinstance(continuation, GlobalAtmosphereContinuation):
        raise TypeError("continuation must be a GlobalAtmosphereContinuation.")
    steps = int(maximum_steps)
    scales = (
        float(flux_tolerance_w_per_m2),
        float(maximum_air_temperature_shift),
        float(maximum_surface_temperature_shift),
        float(maximum_jacobian_condition),
    )
    if steps != maximum_steps or steps < 1:
        raise ValueError("maximum_steps must be a positive integer.")
    if any(not math.isfinite(value) or value <= 0 for value in scales):
        raise ValueError(
            "Flux, shift, and conditioning limits must be positive and finite."
        )
    processes = model.plan.processes
    boundary = processes.surface_physics
    if boundary is None or processes.thermodynamics is None:
        raise ValueError(
            "Flux preconditioning requires the interactive moist surface owner."
        )
    if not bool(model.admissible(continuation.state)):
        raise ValueError("Flux preconditioning requires an admissible initial state.")
    if bool(continuation.accepted_steps != 0) or bool(continuation.rejected_steps != 0):
        raise ValueError(
            "Flux preconditioning is an initial-state preparation, not a restart step."
        )
    base = continuation
    base_view = model.view(base.state)
    base_surface_temperature = boundary.temperature(
        base.state.surface_water,
        base.state.surface_energy,
        processes.thermodynamics,
    )

    def shifted(controls):
        temperature = base_view.temperature + controls[0]
        slab = boundary.slab.initialize(
            base_surface_temperature + controls[1], base.state.surface_water
        )
        state = eqx.tree_at(
            lambda value: (
                value.temperature,
                value.surface_water,
                value.surface_energy,
            ),
            base.state,
            (model.project(temperature), slab.water_mass, slab.energy),
        )
        held = processes.forcing(model.view(state), model.work_space.transform.theta)
        return eqx.tree_at(
            lambda value: (value.state, value.held_forcing, value.forcing_age),
            base,
            (state, held, jnp.asarray(0, jnp.int32)),
        )

    def residual_and_valid(controls):
        return global_flux_residuals(model, shifted(controls))

    controls = jnp.zeros((2,), dtype=base.state.surface_energy.dtype)
    initial_residual, initial_valid = residual_and_valid(controls)
    history = [initial_residual]
    accepted_iterations = 0
    condition = jnp.asarray(jnp.inf, dtype=controls.dtype)
    successful = (
        bool(initial_valid) and float(jnp.max(jnp.abs(initial_residual))) <= scales[0]
    )
    for _ in range(steps):
        if successful:
            break
        residual, valid = residual_and_valid(controls)
        if not bool(valid):
            break
        jacobian = jax.jacfwd(lambda value: residual_and_valid(value)[0])(controls)
        linear_result = la.solve_small_linear(
            la.SmallLinearSolvePlan(2),
            jacobian,
            residual,
        )
        condition = linear_result.condition_estimate
        direction = linear_result.value
        regular = linear_result.successful
        if (
            not bool(regular)
            or not bool(jnp.isfinite(condition))
            or float(condition) > scales[3]
        ):
            break
        current_norm = float(jnp.max(jnp.abs(residual)))
        selected = None
        selected_residual = None
        for damping in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            proposal = controls - damping * direction
            within = (
                abs(float(proposal[0])) <= scales[1]
                and abs(float(proposal[1])) <= scales[2]
            )
            if not within:
                continue
            proposal_residual, proposal_valid = residual_and_valid(proposal)
            if (
                bool(proposal_valid)
                and float(jnp.max(jnp.abs(proposal_residual))) < current_norm
            ):
                selected = proposal
                selected_residual = proposal_residual
                break
        if selected is None:
            break
        controls = selected
        history.append(selected_residual)
        accepted_iterations += 1
        successful = float(jnp.max(jnp.abs(selected_residual))) <= scales[0]

    attempted = shifted(controls)
    final_residual, final_valid = residual_and_valid(controls)
    successful = successful and bool(final_valid)
    if successful:
        result_continuation = attempted
        result_controls = controls
        preparation_energy = (
            model.inventories(attempted.state)[2] - model.inventories(base.state)[2]
        )
    else:
        result_continuation = base
        result_controls = jnp.zeros_like(controls)
        preparation_energy = jnp.asarray(0.0, dtype=controls.dtype)
    result_residual = final_residual if successful else initial_residual
    preparation_id = canonical_fingerprint(
        {
            "kind": "global-two-temperature-flux-preconditioning",
            "model": model.prepared_id,
            "initial_state": array_tree_fingerprint(base.state),
            "maximum_steps": steps,
            "flux_tolerance_w_per_m2": scales[0],
            "maximum_air_temperature_shift": scales[1],
            "maximum_surface_temperature_shift": scales[2],
            "maximum_jacobian_condition": scales[3],
        }
    )
    return GlobalFluxPreconditioningResult(
        result_continuation,
        result_controls[0],
        result_controls[1],
        initial_residual,
        result_residual,
        jnp.stack(history),
        jnp.asarray(accepted_iterations, jnp.int32),
        condition,
        preparation_energy,
        jnp.asarray(successful),
        preparation_id,
    )


__all__ = [
    "GlobalFluxPreconditioningResult",
    "global_flux_residuals",
    "precondition_global_fluxes",
]
