#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._thermal import ThermalLatticeBoltzmannPlan


class SolidThermalEnergyState(StrictModule):
    """Volumetric sensible energy retained as checkpointable solid state."""

    sensible_energy: Array
    successful: Array
    step_index: Array
    plan_id: str = eqx.field(static=True)
    state_id: str = eqx.field(static=True)

    def __init__(
        self,
        sensible_energy: ArrayLike,
        successful: ArrayLike,
        step_index: ArrayLike,
        plan_id: str,
        /,
    ):
        energy = jnp.asarray(sensible_energy)
        success = jnp.asarray(successful, dtype=bool)
        step = jnp.asarray(step_index)
        identifier = str(plan_id)
        if energy.ndim == 0:
            raise ValueError("Solid sensible energy must have at least one spatial axis.")
        if not jnp.issubdtype(energy.dtype, jnp.inexact):
            raise TypeError("Solid sensible energy must use an inexact dtype.")
        if (
            success.shape != ()
            or step.shape != ()
            or not jnp.issubdtype(step.dtype, jnp.integer)
        ):
            raise ValueError(
                "Solid success and step index must be Boolean/integer scalars."
            )
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.sensible_energy = energy
        self.successful = success
        self.step_index = step
        self.plan_id = identifier
        self.state_id = canonical_fingerprint(
            {
                "kind": "solid-thermal-energy-state",
                "plan": identifier,
                "shape": list(energy.shape),
                "dtype": str(energy.dtype),
            }
        )


class ConjugateThermalInterfaceFlux(StrictModule):
    """One-to-one interface exchange with exact equal-and-opposite energy rates."""

    heat_flux: Array
    heat_rate: Array
    fluid_energy_rate: Array
    solid_energy_rate: Array
    conservation_residual: Array
    successful: Array


class ConjugateThermalStepResult(StrictModule):
    """Candidate and accepted fluid/solid states for one interface exchange."""

    candidate_fluid_sensible_energy: Array
    fluid_sensible_energy: Array
    candidate_solid_state: SolidThermalEnergyState
    solid_state: SolidThermalEnergyState
    interface_flux: ConjugateThermalInterfaceFlux
    conservation_residual: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class ConjugateThermalPlan(StrictModule, NonTrainableState):
    """Fluid/solid sensible-energy closure with a resolved resistance interface.

    The model is a passive-energy conjugate coupling. It is deliberately not a
    compressible total-energy model. A positive heat rate leaves the fluid and
    enters the solid with no independently rounded source construction.
    """

    fluid: ThermalLatticeBoltzmannPlan
    solid_volumetric_heat_capacity: Array
    solid_thermal_conductivity: Array
    contact_resistance: Array
    model_label: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fluid: ThermalLatticeBoltzmannPlan,
        solid_volumetric_heat_capacity: ArrayLike,
        solid_thermal_conductivity: ArrayLike,
        /,
        *,
        contact_resistance: ArrayLike = 0.0,
        model_label: str = "passive-sensible-energy-conjugate-thermal",
    ):
        if not isinstance(fluid, ThermalLatticeBoltzmannPlan):
            raise TypeError("fluid must be a ThermalLatticeBoltzmannPlan.")
        capacity = np.asarray(solid_volumetric_heat_capacity, dtype=float)
        conductivity = np.asarray(solid_thermal_conductivity, dtype=float)
        contact = np.asarray(contact_resistance, dtype=float)
        if capacity.shape != () or not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError(
                "solid_volumetric_heat_capacity must be finite and positive."
            )
        if (
            conductivity.shape != ()
            or not np.isfinite(conductivity)
            or conductivity <= 0.0
        ):
            raise ValueError("solid_thermal_conductivity must be finite and positive.")
        if contact.shape != () or not np.isfinite(contact) or contact < 0.0:
            raise ValueError("contact_resistance must be finite and nonnegative.")
        label = str(model_label)
        if not label or label != label.strip():
            raise ValueError("model_label must be a nonempty canonical identifier.")
        self.fluid = fluid
        self.solid_volumetric_heat_capacity = jnp.asarray(capacity)
        self.solid_thermal_conductivity = jnp.asarray(conductivity)
        self.contact_resistance = jnp.asarray(contact)
        self.model_label = label
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conjugate-thermal-plan",
                "fluid": fluid.plan_id,
                "solid_volumetric_heat_capacity": float(capacity),
                "solid_thermal_conductivity": float(conductivity),
                "contact_resistance": float(contact),
                "model_label": label,
            }
        )

    def initialize_solid(
        self,
        temperature: ArrayLike,
        /,
        *,
        step_index: ArrayLike = 0,
    ) -> SolidThermalEnergyState:
        value = jnp.asarray(
            temperature,
            dtype=self.solid_volumetric_heat_capacity.dtype,
        )
        energy = self.solid_volumetric_heat_capacity * (
            value - self.fluid.reference_temperature
        )
        successful = jnp.all(jnp.isfinite(value)) & jnp.all(jnp.isfinite(energy))
        return SolidThermalEnergyState(energy, successful, step_index, self.plan_id)

    def solid_temperature(self, state: SolidThermalEnergyState, /) -> Array:
        if not isinstance(state, SolidThermalEnergyState):
            raise TypeError("state must be SolidThermalEnergyState.")
        if state.plan_id != self.plan_id:
            raise ValueError("Solid state belongs to a different conjugate-thermal plan.")
        return (
            state.sensible_energy
            / self.solid_volumetric_heat_capacity.astype(state.sensible_energy.dtype)
            + self.fluid.reference_temperature
        )

    def prepare(
        self,
        fluid_normal_distance: ArrayLike,
        solid_normal_distance: ArrayLike,
        interface_measure: ArrayLike,
        /,
    ) -> "PreparedConjugateThermalPlan":
        fluid_distance = np.asarray(fluid_normal_distance, dtype=float)
        solid_distance = np.asarray(solid_normal_distance, dtype=float)
        measure = np.asarray(interface_measure, dtype=float)
        if (
            fluid_distance.ndim == 0
            or fluid_distance.shape != solid_distance.shape
            or fluid_distance.shape != measure.shape
        ):
            raise ValueError(
                "Conjugate interface geometry must use matching non-scalar arrays."
            )
        if (
            np.any(~np.isfinite(fluid_distance))
            or np.any(~np.isfinite(solid_distance))
            or np.any(~np.isfinite(measure))
            or np.any(fluid_distance <= 0.0)
            or np.any(solid_distance <= 0.0)
            or np.any(measure <= 0.0)
        ):
            raise ValueError(
                "Conjugate interface distances and measures must be finite and positive."
            )
        resistance = (
            fluid_distance / float(self.fluid.thermal_conductivity)
            + float(self.contact_resistance)
            + solid_distance / float(self.solid_thermal_conductivity)
        )
        conductance = 1.0 / resistance
        geometry_id = canonical_fingerprint(
            {
                "kind": "conjugate-thermal-interface-geometry",
                "fluid_distance": array_tree_fingerprint(fluid_distance),
                "solid_distance": array_tree_fingerprint(solid_distance),
                "interface_measure": array_tree_fingerprint(measure),
            }
        )
        return PreparedConjugateThermalPlan(
            self,
            jnp.asarray(conductance),
            jnp.asarray(measure),
            geometry_id,
        )


class PreparedConjugateThermalPlan(StrictModule, NonTrainableState):
    """Geometry-bound conjugate interface operator with no runtime preparation."""

    plan: ConjugateThermalPlan
    conductance: Array
    interface_measure: Array
    geometry_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ConjugateThermalPlan,
        conductance: Array,
        interface_measure: Array,
        geometry_id: str,
        /,
    ):
        if not isinstance(plan, ConjugateThermalPlan):
            raise TypeError("plan must be ConjugateThermalPlan.")
        conductance_ = jnp.asarray(conductance)
        measure = jnp.asarray(interface_measure, dtype=conductance_.dtype)
        if conductance_.ndim == 0 or conductance_.shape != measure.shape:
            raise ValueError(
                "Prepared interface conductance and measure shapes are invalid."
            )
        identifier = str(geometry_id)
        if not identifier:
            raise ValueError("geometry_id must be nonempty.")
        self.plan = plan
        self.conductance = conductance_
        self.interface_measure = measure
        self.geometry_id = identifier
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-conjugate-thermal-plan",
                "plan": plan.plan_id,
                "geometry": identifier,
            }
        )

    def interface_flux(
        self,
        fluid_sensible_energy: ArrayLike,
        solid_state: SolidThermalEnergyState,
        /,
    ) -> ConjugateThermalInterfaceFlux:
        if not isinstance(solid_state, SolidThermalEnergyState):
            raise TypeError("solid_state must be SolidThermalEnergyState.")
        if solid_state.plan_id != self.plan.plan_id:
            raise ValueError("Solid state belongs to a different conjugate-thermal plan.")
        fluid_energy = jnp.asarray(fluid_sensible_energy)
        if not jnp.issubdtype(fluid_energy.dtype, jnp.inexact):
            raise TypeError("Fluid sensible energy must use an inexact dtype.")
        if (
            fluid_energy.shape != self.conductance.shape
            or solid_state.sensible_energy.shape != self.conductance.shape
        ):
            raise ValueError(
                "Fluid, solid, and prepared interface shapes must match exactly."
            )
        fluid_temperature = (
            fluid_energy
            / self.plan.fluid.volumetric_heat_capacity.astype(fluid_energy.dtype)
            + self.plan.fluid.reference_temperature
        )
        solid_temperature = self.plan.solid_temperature(solid_state)
        heat_flux = self.conductance.astype(fluid_energy.dtype) * (
            fluid_temperature - solid_temperature
        )
        heat_rate = heat_flux * self.interface_measure.astype(fluid_energy.dtype)
        fluid_rate = -heat_rate
        solid_rate = heat_rate
        residual = jnp.sum(fluid_rate + solid_rate)
        successful = (
            solid_state.successful
            & jnp.all(jnp.isfinite(fluid_energy))
            & jnp.all(jnp.isfinite(solid_state.sensible_energy))
            & jnp.all(jnp.isfinite(heat_flux))
            & jnp.isfinite(residual)
        )
        return ConjugateThermalInterfaceFlux(
            heat_flux,
            heat_rate,
            fluid_rate,
            solid_rate,
            residual,
            successful,
        )

    def execute(
        self,
        fluid_sensible_energy: ArrayLike,
        solid_state: SolidThermalEnergyState,
        step_size: ArrayLike,
        fluid_cell_measure: ArrayLike,
        solid_cell_measure: ArrayLike,
        /,
    ) -> ConjugateThermalStepResult:
        flux = self.interface_flux(fluid_sensible_energy, solid_state)
        fluid_energy = jnp.asarray(fluid_sensible_energy)
        dt = jnp.asarray(step_size, dtype=fluid_energy.dtype)
        fluid_measure = jnp.asarray(fluid_cell_measure, dtype=fluid_energy.dtype)
        solid_measure = jnp.asarray(solid_cell_measure, dtype=fluid_energy.dtype)
        if dt.shape != ():
            raise ValueError("Conjugate-thermal step_size must be scalar.")
        if fluid_measure.shape not in (
            (),
            fluid_energy.shape,
        ) or solid_measure.shape not in (
            (),
            fluid_energy.shape,
        ):
            raise ValueError(
                "Conjugate-thermal cell measures must be scalar or interface-shaped."
            )
        positive_measure = jnp.all(fluid_measure > 0.0) & jnp.all(solid_measure > 0.0)
        candidate_fluid = fluid_energy + dt * flux.fluid_energy_rate / fluid_measure
        candidate_solid_energy = (
            solid_state.sensible_energy + dt * flux.solid_energy_rate / solid_measure
        )
        fluid_change = jnp.sum((candidate_fluid - fluid_energy) * fluid_measure)
        solid_change = jnp.sum(
            (candidate_solid_energy - solid_state.sensible_energy) * solid_measure
        )
        conservation_residual = fluid_change + solid_change
        successful = (
            flux.successful
            & jnp.isfinite(dt)
            & (dt >= 0.0)
            & positive_measure
            & jnp.all(jnp.isfinite(candidate_fluid))
            & jnp.all(jnp.isfinite(candidate_solid_energy))
            & jnp.isfinite(conservation_residual)
        )
        candidate_solid = SolidThermalEnergyState(
            candidate_solid_energy,
            successful,
            solid_state.step_index + 1,
            self.plan.plan_id,
        )
        accepted_solid = SolidThermalEnergyState(
            jnp.where(successful, candidate_solid_energy, solid_state.sensible_energy),
            successful,
            jnp.where(successful, solid_state.step_index + 1, solid_state.step_index),
            self.plan.plan_id,
        )
        accepted_fluid = jnp.where(successful, candidate_fluid, fluid_energy)
        return ConjugateThermalStepResult(
            candidate_fluid,
            accepted_fluid,
            candidate_solid,
            accepted_solid,
            flux,
            conservation_residual,
            successful,
            self.prepared_id,
        )


__all__ = [
    "ConjugateThermalInterfaceFlux",
    "ConjugateThermalPlan",
    "ConjugateThermalStepResult",
    "PreparedConjugateThermalPlan",
    "SolidThermalEnergyState",
]
