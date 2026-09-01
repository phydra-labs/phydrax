#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..dynamics import DAEStructure, DifferentialAlgebraicSystem, TimeGrid
from ..equations._chemical_mechanism import PreparedChemicalMechanism
from ..equations._chemical_rates import ChemicalRateRuntime
from ..equations._chemical_species import ChemicalPhaseKind
from ..equations._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT
from ._bdf_method import BDFMethod
from ._differential import DifferentialProblem
from ._differential_algebraic import (
    DAESolvePolicy,
    DifferentialAlgebraicProblem,
    solve_dae,
)
from ._rosenbrock import (
    RosenbrockAdaptivePolicy,
    solve_rosenbrock_adaptive,
)


@jax.custom_jvp
def _implicit_reactor_temperature(
    temperature,
    conserved_energy,
    species_amount,
    molar_energy,
    molar_heat_capacity,
):
    del conserved_energy, species_amount, molar_energy, molar_heat_capacity
    return temperature


@_implicit_reactor_temperature.defjvp
def _implicit_reactor_temperature_jvp(primals, tangents):
    temperature, conserved, amount, molar_energy, molar_capacity = primals
    _, conserved_tangent, amount_tangent, molar_energy_tangent, _ = tangents
    capacity = jnp.sum(amount * molar_capacity)
    temperature_tangent = (
        conserved_tangent
        - jnp.sum(amount_tangent * molar_energy)
        - jnp.sum(amount * molar_energy_tangent)
    ) / jnp.maximum(capacity, jnp.finfo(capacity.dtype).tiny)
    return temperature, temperature_tangent


class ChemicalReactorKind(StrEnum):
    ISOTHERMAL_CONSTANT_VOLUME = "isothermal_constant_volume"
    ADIABATIC_CONSTANT_VOLUME = "adiabatic_constant_volume"
    ISOTHERMAL_CONSTANT_PRESSURE = "isothermal_constant_pressure"
    ADIABATIC_CONSTANT_PRESSURE = "adiabatic_constant_pressure"


class ChemicalReactorThermodynamicState(StrictModule):
    species_amount: Array
    temperature: Array
    pressure: Array
    volume: Array
    conserved_energy: Array
    temperature_residual: Array
    successful: Array


class ChemicalReactorSolution(StrictModule):
    times: Array
    states: Array
    temperature: Array
    pressure: Array
    volume: Array
    valid: Array
    successful: Array
    temporal_solution: Any
    reactor_id: str = eqx.field(static=True)


class ChemicalReactorPlan(StrictModule, NonTrainableState):
    mechanism: PreparedChemicalMechanism
    kind: ChemicalReactorKind = eqx.field(static=True)
    fixed_temperature: float | None = eqx.field(static=True)
    fixed_volume: float | None = eqx.field(static=True)
    fixed_pressure: float | None = eqx.field(static=True)
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    inversion_iterations: int = eqx.field(static=True)
    reactor_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedChemicalMechanism,
        kind: ChemicalReactorKind,
        /,
        *,
        fixed_temperature: float | None = None,
        fixed_volume: float | None = None,
        fixed_pressure: float | None = None,
        minimum_temperature: float | None = None,
        maximum_temperature: float | None = None,
        inversion_iterations: int = 48,
    ):
        if not isinstance(mechanism, PreparedChemicalMechanism):
            raise TypeError("mechanism must be PreparedChemicalMechanism.")
        if not isinstance(kind, ChemicalReactorKind):
            raise TypeError("kind must be ChemicalReactorKind.")
        temperature = None if fixed_temperature is None else float(fixed_temperature)
        volume = None if fixed_volume is None else float(fixed_volume)
        pressure = None if fixed_pressure is None else float(fixed_pressure)
        isothermal = kind in (
            ChemicalReactorKind.ISOTHERMAL_CONSTANT_VOLUME,
            ChemicalReactorKind.ISOTHERMAL_CONSTANT_PRESSURE,
        )
        constant_volume = kind in (
            ChemicalReactorKind.ISOTHERMAL_CONSTANT_VOLUME,
            ChemicalReactorKind.ADIABATIC_CONSTANT_VOLUME,
        )
        if isothermal != (temperature is not None):
            raise ValueError("Exactly isothermal reactors require fixed_temperature.")
        if constant_volume != (volume is not None) or constant_volume == (
            pressure is not None
        ):
            raise ValueError(
                "Reactor kind requires exactly one fixed volume or pressure."
            )
        if temperature is not None and (
            not np.isfinite(temperature) or temperature <= 0.0
        ):
            raise ValueError("fixed_temperature must be finite and positive.")
        if volume is not None and (not np.isfinite(volume) or volume <= 0.0):
            raise ValueError("fixed_volume must be finite and positive.")
        if pressure is not None and (not np.isfinite(pressure) or pressure <= 0.0):
            raise ValueError("fixed_pressure must be finite and positive.")
        t_min = (
            mechanism.thermodynamics.minimum_temperature
            if minimum_temperature is None
            else float(minimum_temperature)
        )
        t_max = (
            mechanism.thermodynamics.maximum_temperature
            if maximum_temperature is None
            else float(maximum_temperature)
        )
        iterations = int(inversion_iterations)
        if not 0.0 < t_min < t_max or iterations <= 0:
            raise ValueError("Temperature inversion controls are invalid.")
        if temperature is not None and not t_min <= temperature <= t_max:
            raise ValueError("fixed_temperature lies outside thermodynamic bounds.")
        self.mechanism = mechanism
        self.kind = kind
        self.fixed_temperature = temperature
        self.fixed_volume = volume
        self.fixed_pressure = pressure
        self.minimum_temperature = t_min
        self.maximum_temperature = t_max
        self.inversion_iterations = iterations
        self.reactor_id = canonical_fingerprint(
            {
                "kind": "chemical-reactor",
                "mechanism": mechanism.mechanism_id,
                "reactor_kind": kind.value,
                "temperature": temperature,
                "volume": volume,
                "pressure": pressure,
                "bounds": [t_min, t_max],
                "iterations": iterations,
            }
        )

    @property
    def adiabatic(self) -> bool:
        return self.kind in (
            ChemicalReactorKind.ADIABATIC_CONSTANT_VOLUME,
            ChemicalReactorKind.ADIABATIC_CONSTANT_PRESSURE,
        )

    @property
    def constant_volume(self) -> bool:
        return self.kind in (
            ChemicalReactorKind.ISOTHERMAL_CONSTANT_VOLUME,
            ChemicalReactorKind.ADIABATIC_CONSTANT_VOLUME,
        )

    def initial_state(
        self,
        species_amount: ArrayLike,
        initial_temperature: ArrayLike | None = None,
        /,
    ) -> Array:
        amount = jnp.asarray(species_amount)
        if amount.shape != (self.mechanism.schema.species_count,):
            raise ValueError("species_amount must have the mechanism species shape.")
        if not self.adiabatic:
            return amount
        if initial_temperature is None:
            raise ValueError("Adiabatic reactor initialization requires temperature.")
        temperature = jnp.asarray(initial_temperature, dtype=amount.dtype)
        if temperature.shape != ():
            raise ValueError("initial_temperature must be scalar.")
        thermo = self.mechanism.thermodynamics.evaluate(temperature)
        molar_energy = (
            thermo.molar_internal_energy
            if self.constant_volume
            else thermo.molar_enthalpy
        )
        conserved = jnp.sum(amount * molar_energy)
        return jnp.concatenate((amount, conserved[None]))

    def evaluate(self, state: ArrayLike, /) -> ChemicalReactorThermodynamicState:
        value = jnp.asarray(state)
        expected = self.mechanism.schema.species_count + int(self.adiabatic)
        if value.shape != (expected,):
            raise ValueError("Chemical reactor state has incompatible shape.")
        amount = value[: self.mechanism.schema.species_count]
        if self.adiabatic:
            conserved = value[-1]
            temperature, residual, thermal_success = self._recover_temperature(
                amount, conserved
            )
        else:
            temperature = jnp.asarray(self.fixed_temperature, dtype=value.dtype)
            thermo = self.mechanism.thermodynamics.evaluate(temperature)
            molar_energy = (
                thermo.molar_internal_energy
                if self.constant_volume
                else thermo.molar_enthalpy
            )
            conserved = jnp.sum(amount * molar_energy)
            residual = jnp.asarray(0.0, dtype=value.dtype)
            thermal_success = thermo.successful
        gas_mask = self.mechanism.schema.phase_mask(ChemicalPhaseKind.GAS).astype(
            value.dtype
        )
        gas_amount = jnp.sum(amount * gas_mask)
        if self.constant_volume:
            volume = jnp.asarray(self.fixed_volume, dtype=value.dtype)
            pressure = gas_amount * UNIVERSAL_GAS_CONSTANT * temperature / volume
        else:
            pressure = jnp.asarray(self.fixed_pressure, dtype=value.dtype)
            volume = gas_amount * UNIVERSAL_GAS_CONSTANT * temperature / pressure
        successful = (
            thermal_success
            & jnp.all(jnp.isfinite(amount) & (amount >= 0.0))
            & (gas_amount > 0.0)
            & jnp.isfinite(volume)
            & (volume > 0.0)
            & jnp.isfinite(pressure)
            & (pressure > 0.0)
        )
        return ChemicalReactorThermodynamicState(
            amount,
            temperature,
            pressure,
            volume,
            conserved,
            residual,
            successful,
        )

    def rate(
        self,
        time: Array,
        state: Array,
        args: ChemicalRateRuntime | None = None,
        /,
    ) -> Array:
        del time
        reactor = self.evaluate(state)
        concentration = reactor.species_amount / reactor.volume
        fields = self.mechanism.evaluate(
            concentration,
            reactor.temperature,
            reactor.pressure,
            runtime=args,
        )
        amount_rate = fields.species_amount_rate * reactor.volume
        successful = reactor.successful & fields.successful
        amount_rate = jnp.where(successful, amount_rate, jnp.nan)
        if self.adiabatic:
            return jnp.concatenate((amount_rate, jnp.zeros((1,), dtype=state.dtype)))
        return amount_rate

    def solve(
        self,
        species_amount: ArrayLike,
        time_grid: TimeGrid,
        /,
        *,
        initial_temperature: ArrayLike | None = None,
        runtime: ChemicalRateRuntime | None = None,
        adaptive: RosenbrockAdaptivePolicy | None = None,
    ) -> ChemicalReactorSolution:
        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be TimeGrid.")
        initial = self.initial_state(species_amount, initial_temperature)
        dynamics = PreparedChemicalReactorDynamics(self)
        problem = DifferentialProblem(
            dynamics,
            initial,
            t0=time_grid.t0,
            t1=time_grid.t1,
            args=runtime,
            problem_id=f"chemical-reactor:{self.reactor_id}",
        )
        temporal = solve_rosenbrock_adaptive(
            problem,
            time_grid,
            adaptive=adaptive,
            args=runtime,
        )
        evaluated = jax.vmap(self.evaluate)(temporal.states)
        successful = temporal.successful & jnp.all(evaluated.successful)
        return ChemicalReactorSolution(
            temporal.times,
            temporal.states,
            evaluated.temperature,
            evaluated.pressure,
            evaluated.volume,
            temporal.valid,
            successful,
            temporal,
            self.reactor_id,
        )

    def solve_bdf(
        self,
        species_amount: ArrayLike,
        time_grid: TimeGrid,
        /,
        *,
        initial_temperature: ArrayLike | None = None,
        runtime: ChemicalRateRuntime | None = None,
        maximum_order: int = 2,
    ) -> ChemicalReactorSolution:
        """Integrate the reactor through the native variable-step BDF substrate."""

        if not isinstance(time_grid, TimeGrid):
            raise TypeError("time_grid must be TimeGrid.")
        initial = self.initial_state(species_amount, initial_temperature)
        residual = _ChemicalReactorODEResidual(self)
        system = DifferentialAlgebraicSystem(
            residual,
            state_shape=initial.shape,
            structure=DAEStructure(
                ("differential",) * initial.size,
                component_axis=-1,
            ),
            system_id=f"chemical-reactor-dae:{self.reactor_id}",
        )
        initial_rate = self.rate(time_grid.t0, initial, runtime)
        problem = DifferentialAlgebraicProblem(
            system,
            initial,
            initial_state_rate=initial_rate,
            args=runtime,
            problem_id=f"chemical-reactor-bdf:{self.reactor_id}",
        )
        temporal = solve_dae(
            problem,
            time_grid,
            policy=DAESolvePolicy(
                method=BDFMethod(maximum_order),
                failure="status",
            ),
        )
        evaluated = jax.vmap(self.evaluate)(temporal.states)
        successful = temporal.successful & jnp.all(evaluated.successful)
        return ChemicalReactorSolution(
            temporal.times,
            temporal.states,
            evaluated.temperature,
            evaluated.pressure,
            evaluated.volume,
            temporal.valid,
            successful,
            temporal,
            self.reactor_id,
        )

    def _recover_temperature(self, amount, conserved):
        low = jnp.asarray(self.minimum_temperature, dtype=amount.dtype)
        high = jnp.asarray(self.maximum_temperature, dtype=amount.dtype)

        def extensive(temperature):
            thermo = self.mechanism.thermodynamics.evaluate(temperature)
            molar = (
                thermo.molar_internal_energy
                if self.constant_volume
                else thermo.molar_enthalpy
            )
            return jnp.sum(amount * molar)

        low_energy = extensive(low)
        high_energy = extensive(high)
        admissible = (
            jnp.isfinite(conserved)
            & (conserved >= low_energy)
            & (conserved <= high_energy)
        )
        target = jnp.where(admissible, conserved, 0.5 * (low_energy + high_energy))

        def iteration(_, bracket):
            lower, upper = bracket
            midpoint = 0.5 * (lower + upper)
            midpoint_energy = extensive(midpoint)
            return (
                jnp.where(midpoint_energy > target, lower, midpoint),
                jnp.where(midpoint_energy > target, midpoint, upper),
            )

        low, high = jax.lax.fori_loop(
            0, self.inversion_iterations, iteration, (low, high)
        )
        raw_temperature = 0.5 * (low + high)
        raw_thermo = self.mechanism.thermodynamics.evaluate(raw_temperature)
        raw_molar_energy = (
            raw_thermo.molar_internal_energy
            if self.constant_volume
            else raw_thermo.molar_enthalpy
        )
        raw_capacity = (
            raw_thermo.molar_heat_capacity_volume
            if self.constant_volume
            else raw_thermo.molar_heat_capacity_pressure
        )
        temperature = _implicit_reactor_temperature(
            raw_temperature,
            conserved,
            amount,
            raw_molar_energy,
            raw_capacity,
        )
        residual = extensive(temperature) - conserved
        thermo = self.mechanism.thermodynamics.evaluate(temperature)
        heat_capacity = jnp.sum(
            amount
            * (
                thermo.molar_heat_capacity_volume
                if self.constant_volume
                else thermo.molar_heat_capacity_pressure
            )
        )
        tolerance = heat_capacity * (high - low) + (
            256.0 * jnp.finfo(amount.dtype).eps * jnp.maximum(jnp.abs(conserved), 1.0)
        )
        successful = admissible & thermo.successful & (jnp.abs(residual) <= tolerance)
        return temperature, residual, successful


class _ChemicalReactorODEResidual(StrictModule):
    plan: ChemicalReactorPlan

    def __init__(self, plan: ChemicalReactorPlan, /):
        self.plan = plan

    def __call__(self, time, state, state_rate, args=None):
        return state_rate - self.plan.rate(time, state, args)


class PreparedChemicalReactorDynamics(StrictModule):
    plan: ChemicalReactorPlan

    def __init__(self, plan: ChemicalReactorPlan, /):
        if not isinstance(plan, ChemicalReactorPlan):
            raise TypeError("plan must be ChemicalReactorPlan.")
        self.plan = plan

    def __call__(self, time, state, args=None):
        return self.plan.rate(time, state, args)


__all__ = [
    "ChemicalReactorKind",
    "ChemicalReactorPlan",
    "ChemicalReactorSolution",
    "ChemicalReactorThermodynamicState",
    "PreparedChemicalReactorDynamics",
]
