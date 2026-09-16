#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ...solver._chemical_equilibrium import (
    ChemicalEquilibriumEnsemble,
    ChemicalEquilibriumPlan,
    ChemicalEquilibriumResult,
)


class EquilibriumJumpEvidence(StrictModule):
    mass_residual: Array
    momentum_residual: Array
    energy_residual: Array
    sonic_residual: Array
    nonlinear_residual_norm: Array
    iteration_count: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class EquilibriumJumpResult(StrictModule):
    upstream_velocity: Array
    downstream_velocity: Array
    downstream_temperature: Array
    downstream_pressure: Array
    downstream_density: Array
    downstream_equilibrium: ChemicalEquilibriumResult
    evidence: EquilibriumJumpEvidence
    branch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _mass(plan: ChemicalEquilibriumPlan, amount: Array, /) -> Array:
    return contract(
        "s,s->",
        amount,
        plan.thermodynamics.schema.molar_masses.astype(amount.dtype),
    )


def _density_and_specific_enthalpy(plan, amount, temperature, pressure):
    state = plan.evaluate_state(amount, temperature, pressure)
    mass = _mass(plan, amount)
    return mass / state.volume, state.enthalpy / mass


def _sound_speed(plan, equilibrium: ChemicalEquilibriumResult):
    amount = equilibrium.species_amount
    total = jnp.sum(amount)
    fraction = amount / total
    species = plan.thermodynamics.thermodynamics.evaluate(equilibrium.temperature)
    cp = jnp.sum(fraction * species.molar_heat_capacity_pressure)
    cv = jnp.sum(fraction * species.molar_heat_capacity_volume)
    molar_mass = _mass(plan, amount) / total
    gamma = cp / cv
    return jnp.sqrt(gamma * UNIVERSAL_GAS_CONSTANT * equilibrium.temperature / molar_mass)


class EquilibriumShockPlan(StrictModule, NonTrainableState):
    equilibrium: ChemicalEquilibriumPlan
    tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    difference_step: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        equilibrium: ChemicalEquilibriumPlan,
        /,
        *,
        tolerance: float = 1.0e-8,
        maximum_steps: int = 12,
        difference_step: float = 1.0e-4,
    ):
        if (
            not isinstance(equilibrium, ChemicalEquilibriumPlan)
            or equilibrium.ensemble is not ChemicalEquilibriumEnsemble.TP
        ):
            raise TypeError("Equilibrium shocks require a TP ChemicalEquilibriumPlan.")
        tolerance_, step = float(tolerance), float(difference_step)
        maximum = int(maximum_steps)
        if (
            not isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not isfinite(step)
            or step <= 0.0
            or maximum <= 0
        ):
            raise ValueError("Shock nonlinear controls are invalid.")
        self.equilibrium = equilibrium
        self.tolerance = tolerance_
        self.maximum_steps = maximum
        self.difference_step = step
        self.plan_id = canonical_fingerprint(
            {
                "kind": "equilibrium-normal-shock",
                "equilibrium": equilibrium.equilibrium_id,
                "tolerance": tolerance_,
                "maximum_steps": maximum,
                "difference_step": step,
            }
        )

    def solve(
        self,
        upstream_temperature: ArrayLike,
        upstream_pressure: ArrayLike,
        upstream_species_amount: ArrayLike,
        upstream_velocity: ArrayLike,
        /,
        *,
        downstream_temperature_guess: ArrayLike,
        downstream_pressure_guess: ArrayLike,
    ) -> EquilibriumJumpResult:
        temperature = jnp.asarray(upstream_temperature)
        pressure = jnp.asarray(upstream_pressure, dtype=temperature.dtype)
        amount = jnp.asarray(upstream_species_amount, dtype=temperature.dtype)
        velocity = jnp.asarray(upstream_velocity, dtype=temperature.dtype)
        if any(value.shape != () for value in (temperature, pressure, velocity)):
            raise ValueError("Shock temperature, pressure, and velocity must be scalar.")
        if amount.shape != (self.equilibrium.thermodynamics.schema.species_count,):
            raise ValueError("Shock composition does not match the equilibrium plan.")
        upstream_density, upstream_enthalpy = _density_and_specific_enthalpy(
            self.equilibrium, amount, temperature, pressure
        )
        mass_flux = upstream_density * velocity
        momentum_total = pressure + upstream_density * velocity**2
        stagnation_enthalpy = upstream_enthalpy + 0.5 * velocity**2
        momentum_scale = jnp.maximum(jnp.abs(momentum_total), 1.0)
        energy_scale = jnp.maximum(jnp.abs(stagnation_enthalpy), 1.0)

        def residual(log_state):
            downstream_temperature = jnp.exp(log_state[0])
            downstream_pressure = jnp.exp(log_state[1])
            equilibrium = self.equilibrium.solve(
                downstream_temperature, downstream_pressure, amount
            )
            density, enthalpy = _density_and_specific_enthalpy(
                self.equilibrium,
                equilibrium.species_amount,
                downstream_temperature,
                downstream_pressure,
            )
            downstream_velocity = mass_flux / density
            return jnp.asarray(
                (
                    (
                        downstream_pressure
                        + density * downstream_velocity**2
                        - momentum_total
                    )
                    / momentum_scale,
                    (enthalpy + 0.5 * downstream_velocity**2 - stagnation_enthalpy)
                    / energy_scale,
                )
            )

        state = jnp.log(
            jnp.asarray(
                (downstream_temperature_guess, downstream_pressure_guess),
                dtype=temperature.dtype,
            )
        )
        step = self.difference_step
        for _ in range(self.maximum_steps):
            value = residual(state)
            columns = []
            for index in range(2):
                direction = jnp.zeros((2,), dtype=state.dtype).at[index].set(step)
                columns.append(
                    (residual(state + direction) - residual(state - direction))
                    / (2.0 * step)
                )
            jacobian = jnp.stack(tuple(columns), axis=-1)
            update = jnp.linalg.solve(jacobian, -value)
            state = state + jnp.clip(update, -0.5, 0.5)
        downstream_temperature, downstream_pressure = jnp.exp(state)
        downstream_equilibrium = self.equilibrium.solve(
            downstream_temperature, downstream_pressure, amount
        )
        downstream_density, downstream_enthalpy = _density_and_specific_enthalpy(
            self.equilibrium,
            downstream_equilibrium.species_amount,
            downstream_temperature,
            downstream_pressure,
        )
        downstream_velocity = mass_flux / downstream_density
        mass_residual = (
            upstream_density * velocity - downstream_density * downstream_velocity
        )
        momentum_residual = (
            downstream_pressure
            + downstream_density * downstream_velocity**2
            - momentum_total
        )
        energy_residual = (
            downstream_enthalpy + 0.5 * downstream_velocity**2 - stagnation_enthalpy
        )
        normalized = residual(state)
        norm = jnp.linalg.norm(normalized)
        finite = jnp.all(jnp.isfinite(normalized)) & jnp.isfinite(downstream_density)
        successful = (
            finite & downstream_equilibrium.evidence.successful & (norm <= self.tolerance)
        )
        branch = canonical_fingerprint(
            {
                "kind": "equilibrium-shock-branch",
                "plan": self.plan_id,
                "upstream_species_count": int(amount.size),
            }
        )
        evidence = EquilibriumJumpEvidence(
            mass_residual,
            momentum_residual,
            energy_residual,
            jnp.asarray(0.0, dtype=norm.dtype),
            norm,
            jnp.asarray(self.maximum_steps, dtype=jnp.int32),
            finite,
            successful,
            self.plan_id,
        )
        return EquilibriumJumpResult(
            velocity,
            downstream_velocity,
            downstream_temperature,
            downstream_pressure,
            downstream_density,
            downstream_equilibrium,
            evidence,
            branch,
            self.plan_id,
        )


class DetonationJumpPlan(StrictModule, NonTrainableState):
    shock: EquilibriumShockPlan
    drive_ratio: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        equilibrium: ChemicalEquilibriumPlan,
        /,
        *,
        drive_ratio: float = 1.0,
        tolerance: float = 1.0e-8,
        maximum_steps: int = 16,
        difference_step: float = 1.0e-4,
    ):
        ratio = float(drive_ratio)
        if not isfinite(ratio) or ratio <= 0.0:
            raise ValueError("drive_ratio must be finite and positive.")
        shock = EquilibriumShockPlan(
            equilibrium,
            tolerance=tolerance,
            maximum_steps=maximum_steps,
            difference_step=difference_step,
        )
        self.shock = shock
        self.drive_ratio = ratio
        self.plan_id = canonical_fingerprint(
            {
                "kind": "equilibrium-detonation-jump",
                "shock": shock.plan_id,
                "drive_ratio": ratio,
            }
        )

    def solve(
        self,
        upstream_temperature: ArrayLike,
        upstream_pressure: ArrayLike,
        upstream_species_amount: ArrayLike,
        /,
        *,
        wave_speed_guess: ArrayLike,
        downstream_temperature_guess: ArrayLike,
        downstream_pressure_guess: ArrayLike,
    ) -> EquilibriumJumpResult:
        temperature = jnp.asarray(upstream_temperature)
        pressure = jnp.asarray(upstream_pressure, dtype=temperature.dtype)
        amount = jnp.asarray(upstream_species_amount, dtype=temperature.dtype)
        upstream_density, upstream_enthalpy = _density_and_specific_enthalpy(
            self.shock.equilibrium, amount, temperature, pressure
        )
        momentum_scale = jnp.maximum(jnp.abs(pressure), 1.0)
        energy_scale = jnp.maximum(jnp.abs(upstream_enthalpy), 1.0)

        def residual(log_state):
            downstream_temperature, downstream_pressure, wave_speed = jnp.exp(log_state)
            equilibrium = self.shock.equilibrium.solve(
                downstream_temperature, downstream_pressure, amount
            )
            downstream_density, downstream_enthalpy = _density_and_specific_enthalpy(
                self.shock.equilibrium,
                equilibrium.species_amount,
                downstream_temperature,
                downstream_pressure,
            )
            downstream_velocity = upstream_density * wave_speed / downstream_density
            sound = _sound_speed(self.shock.equilibrium, equilibrium)
            return jnp.asarray(
                (
                    (
                        downstream_pressure
                        + downstream_density * downstream_velocity**2
                        - pressure
                        - upstream_density * wave_speed**2
                    )
                    / momentum_scale,
                    (
                        downstream_enthalpy
                        + 0.5 * downstream_velocity**2
                        - upstream_enthalpy
                        - 0.5 * wave_speed**2
                    )
                    / energy_scale,
                    (downstream_velocity - self.drive_ratio * sound)
                    / jnp.maximum(sound, 1.0),
                )
            )

        state = jnp.log(
            jnp.asarray(
                (
                    downstream_temperature_guess,
                    downstream_pressure_guess,
                    wave_speed_guess,
                ),
                dtype=temperature.dtype,
            )
        )
        step = self.shock.difference_step
        for _ in range(self.shock.maximum_steps):
            value = residual(state)
            columns = []
            for index in range(3):
                direction = jnp.zeros((3,), dtype=state.dtype).at[index].set(step)
                columns.append(
                    (residual(state + direction) - residual(state - direction))
                    / (2.0 * step)
                )
            jacobian = jnp.stack(tuple(columns), axis=-1)
            update = jnp.linalg.solve(jacobian, -value)
            state = state + jnp.clip(update, -0.5, 0.5)
        downstream_temperature, downstream_pressure, wave_speed = jnp.exp(state)
        equilibrium = self.shock.equilibrium.solve(
            downstream_temperature, downstream_pressure, amount
        )
        downstream_density, downstream_enthalpy = _density_and_specific_enthalpy(
            self.shock.equilibrium,
            equilibrium.species_amount,
            downstream_temperature,
            downstream_pressure,
        )
        downstream_velocity = upstream_density * wave_speed / downstream_density
        sound = _sound_speed(self.shock.equilibrium, equilibrium)
        mass_residual = (
            upstream_density * wave_speed - downstream_density * downstream_velocity
        )
        momentum_residual = (
            downstream_pressure
            + downstream_density * downstream_velocity**2
            - pressure
            - upstream_density * wave_speed**2
        )
        energy_residual = (
            downstream_enthalpy
            + 0.5 * downstream_velocity**2
            - upstream_enthalpy
            - 0.5 * wave_speed**2
        )
        sonic = downstream_velocity - self.drive_ratio * sound
        normalized = residual(state)
        norm = jnp.linalg.norm(normalized)
        finite = jnp.all(jnp.isfinite(normalized))
        successful = (
            finite & equilibrium.evidence.successful & (norm <= self.shock.tolerance)
        )
        branch = canonical_fingerprint(
            {
                "kind": "equilibrium-detonation-branch",
                "plan": self.plan_id,
                "drive_ratio": self.drive_ratio,
            }
        )
        evidence = EquilibriumJumpEvidence(
            mass_residual,
            momentum_residual,
            energy_residual,
            sonic,
            norm,
            jnp.asarray(self.shock.maximum_steps, dtype=jnp.int32),
            finite,
            successful,
            self.plan_id,
        )
        return EquilibriumJumpResult(
            wave_speed,
            downstream_velocity,
            downstream_temperature,
            downstream_pressure,
            downstream_density,
            equilibrium,
            evidence,
            branch,
            self.plan_id,
        )


__all__ = [
    "DetonationJumpPlan",
    "EquilibriumJumpEvidence",
    "EquilibriumJumpResult",
    "EquilibriumShockPlan",
]
