#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...dynamics import TimeGrid
from ...metrix import EuclideanStateGeometry
from ...solver import (
    AbstractFixedStepMethod,
    FixedStepProblem,
    FixedStepResult,
    FixedStepRolloutPlan,
    FixedStepRolloutResult,
)
from ..geophysics._time import GeophysicalTimeSpec
from ._energy import MultilayerEnergyBalance
from ._forcing import ClimateForcingResult, Myhre1998Forcing
from ._gas import GasBoxModel, MODEL_YEAR_SECONDS


class ClimateDrivers(StrictModule):
    """One interval or a leading time axis of interval drivers.

    Emissions are inventory/model-year; concentrations are absolute endpoint
    values; gas_forcing and named external channels are interval-constant
    W m^-2. Inactive gas-role entries are ignored, not silently converted.
    Leading scenario/config/member axes are ordinary JAX vmap axes.
    """

    emissions: Array
    concentrations: Array
    gas_forcing: Array
    external_forcing: Array

    def __init__(
        self,
        emissions: ArrayLike,
        concentrations: ArrayLike,
        gas_forcing: ArrayLike,
        external_forcing: ArrayLike,
    ):
        self.emissions = jnp.asarray(emissions)
        self.concentrations = jnp.asarray(concentrations)
        self.gas_forcing = jnp.asarray(gas_forcing)
        self.external_forcing = jnp.asarray(external_forcing)


class ReducedClimateState(StrictModule):
    boxes: Array
    temperature: Array
    cumulative_emissions: Array
    cumulative_sink: Array
    cumulative_forcing_energy: Array
    cumulative_outgoing_energy: Array
    time: Array
    step_index: Array
    prepared_id: str = eqx.field(static=True)


class ReducedClimateDiagnostics(StrictModule):
    gas_budget_residual: Array
    energy_budget_residual: Array
    lifetime_residual: Array
    lifetime_bracketed: Array
    inferred_emissions: Array
    successful: Array


class ReducedClimateStepResult(StrictModule):
    state: ReducedClimateState
    candidate_state: ReducedClimateState
    forcing: ClimateForcingResult
    diagnostics: ReducedClimateDiagnostics
    successful: Array


class ReducedClimatePlan(StrictModule):
    """Immutable scientific definition; prepare binds one native numeric clock.

    Reservoir maps and the constant-forcing thermal subproblem are exact.
    Lifetime coefficients are frozen at interval entry and the gas forcing
    is endpoint-averaged. The coupled nonlinear model is a discretization,
    generally first order when state-dependent lifetimes are enabled.
    """

    gases: GasBoxModel
    energy: MultilayerEnergyBalance
    forcing: Myhre1998Forcing
    roles: tuple[str, ...] = eqx.field(static=True)
    budget_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        gases: GasBoxModel | None = None,
        energy: MultilayerEnergyBalance | None = None,
        forcing: Myhre1998Forcing | None = None,
        *,
        roles: tuple[str, ...] = ("emissions", "emissions", "emissions"),
        budget_tolerance: float = 1.0e-5,
    ):
        gases_ = GasBoxModel() if gases is None else gases
        energy_ = MultilayerEnergyBalance() if energy is None else energy
        forcing_ = Myhre1998Forcing() if forcing is None else forcing
        if (
            not isinstance(gases_, GasBoxModel)
            or not isinstance(energy_, MultilayerEnergyBalance)
            or not isinstance(forcing_, Myhre1998Forcing)
        ):
            raise TypeError(
                "Reduced climate requires native gas, energy and forcing models."
            )
        roles_ = tuple(roles)
        if len(roles_) != 3 or any(
            role not in ("emissions", "concentration", "forcing") for role in roles_
        ):
            raise ValueError(
                "Exactly one emissions/concentration/forcing role is required for each gas."
            )
        if not np.isfinite(budget_tolerance) or budget_tolerance <= 0.0:
            raise ValueError("Budget tolerance must be finite and positive.")
        self.gases, self.energy, self.forcing = gases_, energy_, forcing_
        self.roles = roles_
        self.budget_tolerance = float(budget_tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reduced-climate-plan",
                "gases": gases_.model_id,
                "energy": energy_.model_id,
                "forcing": forcing_.forcing_id,
                "roles": roles_,
                "budget_tolerance": self.budget_tolerance,
                "coupling": "frozen-entry-lifetime-endpoint-mean-forcing",
            }
        )

    def prepare(
        self,
        grid: TimeGrid,
        /,
        *,
        time_spec: GeophysicalTimeSpec | None = None,
        seconds_per_time_unit: float | None = None,
    ) -> "PreparedReducedClimate":
        return PreparedReducedClimate(
            self, grid, time_spec=time_spec, seconds_per_time_unit=seconds_per_time_unit
        )


class PreparedReducedClimate(StrictModule):
    plan: ReducedClimatePlan
    start_time: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    seconds_per_time_unit: float = eqx.field(static=True)
    time_id: str = eqx.field(static=True)
    clock_id: str | None = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ReducedClimatePlan,
        grid: TimeGrid,
        /,
        *,
        time_spec: GeophysicalTimeSpec | None = None,
        seconds_per_time_unit: float | None = None,
    ):
        if not isinstance(plan, ReducedClimatePlan) or not isinstance(grid, TimeGrid):
            raise TypeError(
                "Preparation requires ReducedClimatePlan and native TimeGrid."
            )
        if (time_spec is None) == (seconds_per_time_unit is None):
            raise ValueError(
                "Bind exactly one geophysical time spec or explicit seconds_per_time_unit."
            )
        if time_spec is not None and not isinstance(time_spec, GeophysicalTimeSpec):
            raise TypeError("time_spec must be GeophysicalTimeSpec.")
        scale = float(
            seconds_per_time_unit if time_spec is None else time_spec.seconds_per_unit
        )
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "The numerical clock requires a positive finite seconds conversion."
            )
        times = np.asarray(grid.times, dtype=float)
        durations = np.diff(times)
        if not np.allclose(durations, durations[0], rtol=1.0e-6, atol=1.0e-12):
            raise ValueError(
                "Fixed climate rollout needs a uniform numerical grid; encode calendars separately."
            )
        self.plan = plan
        self.start_time = float(times[0])
        self.step_size = float((times[-1] - times[0]) / grid.num_steps)
        self.step_count = grid.num_steps
        self.seconds_per_time_unit = scale
        self.time_id = grid.time_id
        self.clock_id = None if time_spec is None else time_spec.time_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-reduced-climate",
                "plan": plan.plan_id,
                "numerical_parameters": array_tree_fingerprint(plan),
                "grid": grid.time_id,
                "times": times.tolist(),
                "clock": self.clock_id,
                "seconds_per_time_unit": scale,
            }
        )

    def initial_state(
        self, *, boxes: ArrayLike | None = None, temperature: ArrayLike | None = None
    ) -> ReducedClimateState:
        gases, energy = self.plan.gases, self.plan.energy
        boxes_ = (
            jnp.zeros_like(gases.fractions)
            if boxes is None
            else jnp.asarray(boxes, dtype=gases.fractions.dtype)
        )
        temperature_ = (
            jnp.zeros_like(energy.capacities)
            if temperature is None
            else jnp.asarray(temperature, dtype=energy.capacities.dtype)
        )
        if (
            boxes_.shape != gases.fractions.shape
            or temperature_.shape != energy.capacities.shape
        ):
            raise ValueError(
                "Initial inventories and temperatures must match the prepared model."
            )
        if (
            not np.all(np.isfinite(np.asarray(boxes_)))
            or not np.all(np.isfinite(np.asarray(temperature_)))
            or not np.all(np.asarray(gases.concentration(boxes_)) > 0.0)
        ):
            raise ValueError(
                "Initial state must be finite with positive absolute gas concentrations."
            )
        zero = jnp.zeros((), dtype=boxes_.dtype)
        return ReducedClimateState(
            boxes_,
            temperature_,
            jnp.zeros_like(gases.background),
            jnp.zeros_like(gases.background),
            zero,
            zero,
            jnp.asarray(self.start_time, dtype=boxes_.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self.prepared_id,
        )

    def step(
        self, state: ReducedClimateState, drivers: ClimateDrivers, /
    ) -> ReducedClimateStepResult:
        if state.prepared_id != self.prepared_id:
            raise ValueError("Climate state belongs to another prepared runtime.")
        if drivers.emissions.shape != (3,) or drivers.concentrations.shape != (3,):
            raise ValueError("A climate step requires one three-gas driver vector.")
        plan = self.plan
        dt = jnp.asarray(
            self.step_size * self.seconds_per_time_unit / MODEL_YEAR_SECONDS,
            dtype=state.boxes.dtype,
        )
        gas = plan.gases.advance(
            state.boxes,
            state.cumulative_sink,
            state.temperature[0],
            dt,
            drivers.emissions,
            drivers.concentrations,
            plan.roles,
        )
        incoming_forcing = plan.forcing.evaluate(
            plan.gases.concentration(state.boxes),
            plan.gases.background,
            drivers.external_forcing,
            drivers.gas_forcing,
            plan.roles,
        )
        outgoing_forcing = plan.forcing.evaluate(
            plan.gases.concentration(gas.boxes),
            plan.gases.background,
            drivers.external_forcing,
            drivers.gas_forcing,
            plan.roles,
        )
        components = (incoming_forcing.components + outgoing_forcing.components) * 0.5
        forcing = ClimateForcingResult(
            components,
            jnp.sum(components),
            incoming_forcing.successful & outgoing_forcing.successful,
            incoming_forcing.names,
        )
        thermal = plan.energy.advance(state.temperature, dt, forcing.total)
        candidate = ReducedClimateState(
            gas.boxes,
            thermal.temperature,
            state.cumulative_emissions + gas.emissions * dt,
            state.cumulative_sink + gas.sink_increment,
            state.cumulative_forcing_energy + thermal.input_energy,
            state.cumulative_outgoing_energy + thermal.outgoing_energy,
            state.time + self.step_size,
            state.step_index + 1,
            self.prepared_id,
        )
        gas_residual = (
            jnp.sum(candidate.boxes - state.boxes, axis=-1)
            + gas.sink_increment
            - gas.emissions * dt
        )
        gas_scale = (
            1.0 + jnp.sum(jnp.abs(state.boxes), axis=-1) + jnp.abs(gas.emissions * dt)
        )
        energy_scale = (
            1.0
            + jnp.abs(thermal.input_energy)
            + jnp.abs(thermal.outgoing_energy)
            + jnp.abs(plan.energy.heat_content(state.temperature))
        )
        budgets_ok = jnp.all(
            jnp.abs(gas_residual) <= plan.budget_tolerance * gas_scale
        ) & (jnp.abs(thermal.energy_residual) <= plan.budget_tolerance * energy_scale)
        clock_ok = (
            (state.step_index >= 0)
            & (state.step_index < self.step_count)
            & jnp.isclose(
                state.time,
                self.start_time + state.step_index * self.step_size,
                rtol=0.0,
                atol=1.0e-5 * self.step_size,
            )
        )
        finite_state = jnp.all(
            jnp.stack(
                tuple(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(candidate))
            )
        )
        successful = (
            gas.successful
            & forcing.successful
            & thermal.successful
            & budgets_ok
            & clock_ok
            & finite_state
        )
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        diagnostics = ReducedClimateDiagnostics(
            gas_residual,
            thermal.energy_residual,
            gas.lifetime.residual,
            gas.lifetime.bracketed,
            gas.emissions,
            successful,
        )
        return ReducedClimateStepResult(
            accepted, candidate, forcing, diagnostics, successful
        )

    def problem(
        self,
        initial_state: ReducedClimateState,
        drivers: ClimateDrivers,
        /,
        *,
        start_step: int = 0,
        stop_step: int | None = None,
    ) -> FixedStepProblem:
        """Build a native fixed problem for a full run or an exact restart window.

        Drivers always cover the complete bound grid; start/stop select a
        window without resampling or changing the prepared/checkpoint identity.
        Build outside transformed code, then map dynamic leaves with JAX/Equinox.
        """
        stop = self.step_count if stop_step is None else int(stop_step)
        start = int(start_step)
        if not 0 <= start < stop <= self.step_count:
            raise ValueError("Climate rollout window must lie inside the prepared grid.")
        if initial_state.prepared_id != self.prepared_id:
            raise ValueError("Climate initial state belongs to another runtime.")
        if int(initial_state.step_index) != start or not np.isclose(
            float(initial_state.time),
            self.start_time + start * self.step_size,
            rtol=0.0,
            atol=1.0e-5 * self.step_size,
        ):
            raise ValueError(
                "Restart state must be exactly at the requested window boundary."
            )
        shape = (self.step_count, 3)
        if (
            drivers.emissions.shape != shape
            or drivers.concentrations.shape != shape
            or drivers.gas_forcing.shape != shape
            or drivers.external_forcing.shape
            != (self.step_count, len(self.plan.forcing.external_names))
        ):
            raise ValueError("Driver arrays must cover the complete prepared time grid.")
        return FixedStepProblem(
            ReducedClimateFixedStepMethod(self, start),
            initial_state,
            t0=self.start_time + start * self.step_size,
            t1=self.start_time + stop * self.step_size,
            step_size=self.step_size,
            args=drivers,
            state_geometry=EuclideanStateGeometry(),
        )

    def rollout(
        self,
        initial_state: ReducedClimateState,
        drivers: ClimateDrivers,
        /,
        *,
        start_step: int = 0,
        stop_step: int | None = None,
        retention: str = "trajectory",
    ) -> FixedStepRolloutResult:
        return FixedStepRolloutPlan(retention=retention).rollout(
            self.problem(
                initial_state, drivers, start_step=start_step, stop_step=stop_step
            )
        )


class ReducedClimateFixedStepMethod(AbstractFixedStepMethod):
    climate: PreparedReducedClimate
    start_step: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, climate: PreparedReducedClimate, start_step: int = 0, /):
        self.climate = climate
        self.start_step = int(start_step)
        self.method_id = canonical_fingerprint(
            {
                "kind": "reduced-climate-fixed-step",
                "prepared": climate.prepared_id,
                "start_step": self.start_step,
            }
        )

    @property
    def required_step_size(self) -> float:
        return self.climate.step_size

    @property
    def allows_step_reduction(self) -> bool:
        return False

    def step(
        self,
        step_index: Array,
        time: Array,
        state: ReducedClimateState,
        step_size: Array,
        args: Any,
        /,
    ) -> FixedStepResult:
        drivers = jax.tree.map(lambda values: values[step_index + self.start_step], args)
        result = self.climate.step(state, drivers)
        aligned = jnp.isclose(
            time, state.time, rtol=0.0, atol=1.0e-5 * self.climate.step_size
        ) & (step_size == self.climate.step_size)
        successful = result.successful & aligned
        accepted = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), result.state, state
        )
        residual = jnp.maximum(
            jnp.max(result.diagnostics.lifetime_residual),
            jnp.maximum(
                jnp.max(jnp.abs(result.diagnostics.gas_budget_residual)),
                jnp.abs(result.diagnostics.energy_budget_residual),
            ),
        )
        response_active = any(
            enabled and role != "forcing"
            for enabled, role in zip(
                self.climate.plan.gases.response_active,
                self.climate.plan.roles,
                strict=True,
            )
        )
        iterations = self.climate.plan.gases.solve_iterations if response_active else 0
        return FixedStepResult(
            result.candidate_state,
            accepted,
            successful,
            residual,
            jnp.asarray(iterations, dtype=jnp.int32),
            jnp.asarray(iterations + 2, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.time.dtype),
        )


__all__ = [
    "ClimateDrivers",
    "PreparedReducedClimate",
    "ReducedClimateDiagnostics",
    "ReducedClimateFixedStepMethod",
    "ReducedClimatePlan",
    "ReducedClimateState",
    "ReducedClimateStepResult",
]
