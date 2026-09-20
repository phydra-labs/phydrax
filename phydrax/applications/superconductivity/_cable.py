#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._superconducting_material import SuperconductingMaterialLawPlan


class SuperconductingCableState(StrictModule):
    current: Array
    superconducting_current: Array
    stabilizer_current: Array
    solid_temperature: Array
    coolant_temperature: Array
    protection_active: Array
    time: Array
    accepted_step: Array
    plan_id: str = eqx.field(static=True)


class SuperconductingCableEvidence(StrictModule):
    current_closure_residual: Array
    electrical_energy_residual: Array
    thermal_energy_residual: Array
    minimum_current_sharing_margin: Array
    maximum_temperature: Array
    quench_active: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class SuperconductingCableStepResult(StrictModule):
    candidate: SuperconductingCableState
    accepted: SuperconductingCableState
    electric_field: Array
    joule_power: Array
    dump_power: Array
    evidence: SuperconductingCableEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


class SuperconductingCablePlan(StrictModule, NonTrainableState):
    material: SuperconductingMaterialLawPlan
    cell_lengths: Array
    magnetic_field: Array
    field_angle: Array
    superconductor_area: float = eqx.field(static=True)
    stabilizer_area: float = eqx.field(static=True)
    inductance: float = eqx.field(static=True)
    solid_heat_capacity: Array
    coolant_heat_capacity: Array
    axial_thermal_conductance: float = eqx.field(static=True)
    coolant_heat_transfer_per_length: float = eqx.field(static=True)
    coolant_mass_flow_heat_capacity: float = eqx.field(static=True)
    coolant_inlet_temperature: float = eqx.field(static=True)
    dump_resistance: float = eqx.field(static=True)
    protection_trigger_temperature: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        material: SuperconductingMaterialLawPlan,
        cell_lengths: ArrayLike,
        magnetic_field: ArrayLike,
        field_angle: ArrayLike,
        solid_heat_capacity: ArrayLike,
        coolant_heat_capacity: ArrayLike,
        /,
        *,
        superconductor_area: float,
        stabilizer_area: float,
        inductance: float,
        axial_thermal_conductance: float,
        coolant_heat_transfer_per_length: float,
        coolant_mass_flow_heat_capacity: float,
        coolant_inlet_temperature: float,
        dump_resistance: float,
        protection_trigger_temperature: float,
        tolerance: float = 1.0e-8,
    ):
        if not isinstance(material, SuperconductingMaterialLawPlan):
            raise TypeError("material must be SuperconductingMaterialLawPlan.")
        lengths = np.asarray(cell_lengths, dtype=np.float64)
        field = np.asarray(magnetic_field, dtype=np.float64)
        angle = np.asarray(field_angle, dtype=np.float64)
        solid_capacity = np.asarray(solid_heat_capacity, dtype=np.float64)
        coolant_capacity = np.asarray(coolant_heat_capacity, dtype=np.float64)
        scalars = tuple(
            float(value)
            for value in (
                superconductor_area,
                stabilizer_area,
                inductance,
                axial_thermal_conductance,
                coolant_heat_transfer_per_length,
                coolant_mass_flow_heat_capacity,
                coolant_inlet_temperature,
                dump_resistance,
                protection_trigger_temperature,
                tolerance,
            )
        )
        shape = lengths.shape
        if (
            lengths.ndim != 1
            or lengths.size < 1
            or field.shape != shape
            or angle.shape != shape
            or solid_capacity.shape != shape
            or coolant_capacity.shape != shape
            or any(
                np.any(~np.isfinite(value))
                for value in (lengths, field, angle, solid_capacity, coolant_capacity)
            )
            or np.any(lengths <= 0.0)
            or np.any(field < 0.0)
            or np.any(solid_capacity <= 0.0)
            or np.any(coolant_capacity <= 0.0)
            or any(not isfinite(value) for value in scalars)
            or min(scalars[0], scalars[1], scalars[2], scalars[6], scalars[8], scalars[9])
            <= 0.0
            or min(scalars[3], scalars[4], scalars[5], scalars[7]) < 0.0
        ):
            raise ValueError(
                "Cable geometry, fields, capacities, or parameters are invalid."
            )
        self.material = material
        self.cell_lengths = jnp.asarray(lengths)
        self.magnetic_field = jnp.asarray(field)
        self.field_angle = jnp.asarray(angle)
        self.superconductor_area = scalars[0]
        self.stabilizer_area = scalars[1]
        self.inductance = scalars[2]
        self.axial_thermal_conductance = scalars[3]
        self.coolant_heat_transfer_per_length = scalars[4]
        self.coolant_mass_flow_heat_capacity = scalars[5]
        self.coolant_inlet_temperature = scalars[6]
        self.dump_resistance = scalars[7]
        self.protection_trigger_temperature = scalars[8]
        self.tolerance = scalars[9]
        self.solid_heat_capacity = jnp.asarray(solid_capacity)
        self.coolant_heat_capacity = jnp.asarray(coolant_capacity)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "one-dimensional-superconducting-cable-quench",
                "material": material.material_id,
                "lengths": array_tree_fingerprint(lengths),
                "field": array_tree_fingerprint(field),
                "angle": array_tree_fingerprint(angle),
                "areas": list(scalars[:2]),
                "inductance": scalars[2],
                "thermal": list(scalars[3:7]),
                "protection": list(scalars[7:9]),
                "tolerance": scalars[9],
            }
        )

    @property
    def cell_count(self) -> int:
        return self.cell_lengths.size

    def _sharing(self, current: Array, temperature: Array, /):
        evaluated = self.material.evaluate(
            temperature,
            self.magnetic_field,
            self.field_angle,
            jnp.zeros_like(temperature),
        )
        critical_current = evaluated.critical_current_density * self.superconductor_area
        magnitude = jnp.abs(current)
        upper = jnp.minimum(magnitude, critical_current)
        lower = jnp.zeros_like(upper)

        def iteration(_, bounds):
            low, high = bounds
            superconducting = 0.5 * (low + high)
            safe_critical = jnp.maximum(
                critical_current, jnp.finfo(temperature.dtype).tiny
            )
            super_field = (
                self.material.criterion_electric_field
                * (superconducting / safe_critical) ** self.material.power_law_exponent
            )
            stabilizer_field = (
                self.material.stabilizer_resistivity
                * (magnitude - superconducting)
                / self.stabilizer_area
            )
            increase_super = super_field < stabilizer_field
            return (
                jnp.where(increase_super, superconducting, low),
                jnp.where(increase_super, high, superconducting),
            )

        low, high = jax.lax.fori_loop(0, 48, iteration, (lower, upper))
        superconducting = 0.5 * (low + high)
        superconducting = jnp.where(critical_current > 0.0, superconducting, 0.0)
        stabilizer = magnitude - superconducting
        electric = (
            self.material.stabilizer_resistivity * stabilizer / self.stabilizer_area
        )
        sign = jnp.sign(current)
        return (
            sign * superconducting,
            sign * stabilizer,
            sign * electric,
            critical_current - magnitude,
            evaluated.successful,
        )

    def initialize(
        self,
        current: ArrayLike,
        solid_temperature: ArrayLike,
        coolant_temperature: ArrayLike,
        /,
    ) -> SuperconductingCableState:
        current_ = jnp.asarray(current, dtype=self.cell_lengths.dtype)
        solid = jnp.asarray(solid_temperature, dtype=self.cell_lengths.dtype)
        coolant = jnp.asarray(coolant_temperature, dtype=self.cell_lengths.dtype)
        if (
            current_.shape != ()
            or solid.shape != (self.cell_count,)
            or coolant.shape != solid.shape
        ):
            raise ValueError("Cable current or temperature shapes are invalid.")
        superconducting, stabilizer, _, _, support = self._sharing(current_, solid)
        if not bool(jnp.all(support)):
            raise ValueError("Initial cable state lies outside material support.")
        return SuperconductingCableState(
            current_,
            superconducting,
            stabilizer,
            solid,
            coolant,
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=current_.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def advance(
        self,
        state: SuperconductingCableState,
        step_size: ArrayLike,
        applied_voltage: ArrayLike,
        /,
    ) -> SuperconductingCableStepResult:
        if (
            not isinstance(state, SuperconductingCableState)
            or state.plan_id != self.plan_id
        ):
            raise TypeError("state must belong to this SuperconductingCablePlan.")
        step = jnp.asarray(step_size, dtype=state.current.dtype)
        voltage = jnp.asarray(applied_voltage, dtype=state.current.dtype)
        if step.shape != () or voltage.shape != ():
            raise ValueError("Cable step and applied voltage must be scalar.")
        dump_resistance = jnp.where(state.protection_active, self.dump_resistance, 0.0)
        current = state.current
        for _ in range(6):
            _, _, electric, _, _ = self._sharing(current, state.solid_temperature)
            internal_voltage = jnp.sum(electric * self.cell_lengths)
            resistance = jnp.where(
                jnp.abs(current) > 0.0,
                jnp.abs(internal_voltage / current),
                0.0,
            )
            current = (self.inductance * state.current + step * voltage) / (
                self.inductance + step * (resistance + dump_resistance)
            )
        _, _, electric, _, support = self._sharing(current, state.solid_temperature)
        cell_joule = jnp.abs(electric * current) * self.cell_lengths
        dump_power = dump_resistance * current**2
        conduction = jnp.zeros_like(state.solid_temperature)
        if self.cell_count > 1:
            interface = self.axial_thermal_conductance * (
                state.solid_temperature[1:] - state.solid_temperature[:-1]
            )
            conduction = conduction.at[:-1].add(interface)
            conduction = conduction.at[1:].add(-interface)
        transfer = (
            self.coolant_heat_transfer_per_length
            * self.cell_lengths
            * (state.solid_temperature - state.coolant_temperature)
        )
        solid_rate = (cell_joule + conduction - transfer) / self.solid_heat_capacity
        incoming = jnp.concatenate(
            (
                jnp.asarray((self.coolant_inlet_temperature,), dtype=current.dtype),
                state.coolant_temperature[:-1],
            )
        )
        advection = self.coolant_mass_flow_heat_capacity * (
            incoming - state.coolant_temperature
        )
        coolant_rate = (transfer + advection) / self.coolant_heat_capacity
        solid_candidate = state.solid_temperature + step * solid_rate
        coolant_candidate = state.coolant_temperature + step * coolant_rate
        protection = state.protection_active | (
            jnp.max(solid_candidate) >= self.protection_trigger_temperature
        )
        super_candidate, stabilizer_candidate, _, final_margin, final_support = (
            self._sharing(current, solid_candidate)
        )
        candidate = SuperconductingCableState(
            current,
            super_candidate,
            stabilizer_candidate,
            solid_candidate,
            coolant_candidate,
            protection,
            state.time + step,
            state.accepted_step + 1,
            self.plan_id,
        )
        initial_magnetic = 0.5 * self.inductance * state.current**2
        final_magnetic = 0.5 * self.inductance * current**2
        supplied = step * voltage * current
        internal_loss = step * jnp.sum(cell_joule)
        dump_loss = step * dump_power
        increment = current - state.current
        numerical_dissipation = 0.5 * self.inductance * increment**2
        electrical_residual = (
            final_magnetic
            - initial_magnetic
            - supplied
            + internal_loss
            + dump_loss
            + numerical_dissipation
        )
        thermal_change = jnp.sum(
            self.solid_heat_capacity * (solid_candidate - state.solid_temperature)
            + self.coolant_heat_capacity * (coolant_candidate - state.coolant_temperature)
        )
        advective_exchange = (
            step
            * self.coolant_mass_flow_heat_capacity
            * (self.coolant_inlet_temperature - state.coolant_temperature[-1])
        )
        thermal_residual = thermal_change - internal_loss - advective_exchange
        current_residual = current - (super_candidate + stabilizer_candidate)
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(supplied), jnp.abs(initial_magnetic)), 1.0
        )
        finite = (
            jnp.isfinite(step)
            & (step > 0.0)
            & jnp.isfinite(voltage)
            & jnp.all(jnp.isfinite(solid_candidate))
            & jnp.all(jnp.isfinite(coolant_candidate))
            & jnp.isfinite(electrical_residual)
            & jnp.isfinite(thermal_residual)
        )
        successful = (
            finite
            & jnp.all(support)
            & jnp.all(final_support)
            & jnp.all(solid_candidate > 0.0)
            & jnp.all(coolant_candidate > 0.0)
            & (
                jnp.max(jnp.abs(current_residual))
                <= self.tolerance * jnp.maximum(jnp.abs(current), 1.0)
            )
            & (jnp.abs(electrical_residual) <= self.tolerance * scale)
            & (jnp.abs(thermal_residual) <= self.tolerance * scale)
        )
        accepted = jax.tree.map(
            lambda proposed, prior: jnp.where(successful, proposed, prior),
            candidate,
            state,
        )
        evidence = SuperconductingCableEvidence(
            current_residual,
            electrical_residual,
            thermal_residual,
            jnp.min(final_margin),
            jnp.max(solid_candidate),
            protection,
            finite,
            successful,
            self.plan_id,
        )
        return SuperconductingCableStepResult(
            candidate,
            accepted,
            electric,
            cell_joule,
            dump_power,
            evidence,
            successful,
            self.plan_id,
        )


__all__ = [
    "SuperconductingCableEvidence",
    "SuperconductingCablePlan",
    "SuperconductingCableState",
    "SuperconductingCableStepResult",
]
