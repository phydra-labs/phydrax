#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.finite_volume._incompressible import FaceVelocity
from ...discretization.finite_volume._mac_scalar import PreparedMACScalarTransport
from ...equations._mixture_transport import MixtureAveragedTransportPlan
from ...solver._mac_variable_density import (
    MACVariableDensityProjectionPlan,
    MACVariableDensityProjectionResult,
)
from ._low_mach import LowMachReactingFormulation
from ._transport_runtime import (
    TransportPropertyReuseCandidate,
    TransportPropertyReusePlan,
    TransportPropertyReuseState,
)


class LowMachPressureMode(StrEnum):
    CONSTANT = "constant"
    PRESCRIBED = "prescribed"
    CLOSED = "closed"


class LowMachReactingSDCPlan(StrictModule, NonTrainableState):
    correction_sweeps: int = eqx.field(static=True)
    nonlinear_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, correction_sweeps: int = 2, nonlinear_tolerance: float = 1.0e-8, /
    ):
        sweeps = int(correction_sweeps)
        tolerance = float(nonlinear_tolerance)
        if sweeps < 1 or not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("SDC sweeps and nonlinear_tolerance are invalid.")
        self.correction_sweeps = sweeps
        self.nonlinear_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "two-node-reacting-sdc",
                "correction_sweeps": sweeps,
                "nonlinear_tolerance": tolerance,
            }
        )


class LowMachReactingFlowState(StrictModule):
    velocity: FaceVelocity
    species_density: Array
    enthalpy_density: Array
    thermodynamic_pressure: Array
    mechanical_pressure: Array
    transport_reuse: TransportPropertyReuseState | None
    time: Array
    accepted_step: Array
    plan_id: str = eqx.field(static=True)


class LowMachReactingStepDiagnostics(StrictModule):
    element_defect: Array
    charge_defect: Array
    mass_defect: Array
    enthalpy_defect: Array
    eos_pressure_defect: Array
    divergence_defect: Array
    sdc_residual: Array
    minimum_species_density: Array
    finite: Array
    admissible: Array
    conservative: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class LowMachReactingStepResult(StrictModule):
    candidate: LowMachReactingFlowState
    accepted: LowMachReactingFlowState
    projection: MACVariableDensityProjectionResult
    diagnostics: LowMachReactingStepDiagnostics
    successful: Array
    plan_id: str = eqx.field(static=True)


class _LowMachRates(StrictModule):
    species_density: Array
    enthalpy_density: Array
    divergence_source: Array
    temperature: Array
    density: Array
    pressure: Array
    transport_reuse: TransportPropertyReuseCandidate | None
    successful: Array


class LowMachReactingFlowPlan(StrictModule, NonTrainableState):
    formulation: LowMachReactingFormulation
    mixture_transport: MixtureAveragedTransportPlan
    scalar_transport: PreparedMACScalarTransport
    projection: MACVariableDensityProjectionPlan
    sdc: LowMachReactingSDCPlan
    transport_reuse: TransportPropertyReusePlan | None
    pressure_mode: LowMachPressureMode = eqx.field(static=True)
    species_field_names: tuple[str, ...] = eqx.field(static=True)
    enthalpy_field_name: str = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    eos_tolerance: float = eqx.field(static=True)
    maximum_temperature_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        formulation: LowMachReactingFormulation,
        mixture_transport: MixtureAveragedTransportPlan,
        scalar_transport: PreparedMACScalarTransport,
        projection: MACVariableDensityProjectionPlan,
        sdc: LowMachReactingSDCPlan,
        /,
        *,
        pressure_mode: LowMachPressureMode | str = LowMachPressureMode.CONSTANT,
        transport_reuse: TransportPropertyReusePlan | None = None,
        conservation_tolerance: float = 1.0e-8,
        eos_tolerance: float = 1.0e-7,
        maximum_temperature_iterations: int = 64,
    ):
        if not isinstance(formulation, LowMachReactingFormulation):
            raise TypeError("formulation must be LowMachReactingFormulation.")
        if formulation.mechanism is None:
            raise ValueError("Spatial low-Mach reacting flow requires a mechanism.")
        if not isinstance(mixture_transport, MixtureAveragedTransportPlan):
            raise TypeError("mixture_transport must be MixtureAveragedTransportPlan.")
        if (
            mixture_transport.thermodynamics.model_id
            != formulation.thermodynamics.model_id
        ):
            raise ValueError("Low-Mach formulation and transport thermodynamics differ.")
        if not isinstance(scalar_transport, PreparedMACScalarTransport):
            raise TypeError("scalar_transport must be PreparedMACScalarTransport.")
        if not isinstance(projection, MACVariableDensityProjectionPlan):
            raise TypeError("projection must be MACVariableDensityProjectionPlan.")
        if not isinstance(sdc, LowMachReactingSDCPlan):
            raise TypeError("sdc must be LowMachReactingSDCPlan.")
        if (
            scalar_transport.layout.operators.prepared_id
            != projection.operators.prepared_id
        ):
            raise ValueError("Scalar transport and projection must share MAC operators.")
        if any(
            not axis.periodic
            for axis in projection.operators.discretization.grid.structured_axes
        ):
            raise ValueError(
                "The initial low-Mach mixture-diffusion profile requires periodic axes."
            )
        species_names = tuple(
            f"rho:{name}" for name in formulation.thermodynamics.schema.species_names
        )
        enthalpy_name = "rhoh"
        expected = tuple(sorted((*species_names, enthalpy_name)))
        if scalar_transport.layout.field_names != expected:
            raise ValueError("Scalar transport fields must be rho:<species> plus rhoh.")
        if scalar_transport.problem.reaction is not None or any(
            declaration.source is not None
            or bool(np.any(np.asarray(declaration.diffusivity) != 0.0))
            for declaration in scalar_transport.problem.transports
        ):
            raise ValueError(
                "Low-Mach scalar transport owns advection only; mixture diffusion and chemistry have separate owners."
            )
        if transport_reuse is not None and (
            not isinstance(transport_reuse, TransportPropertyReusePlan)
            or transport_reuse.properties.property_id
            != mixture_transport.properties.property_id
        ):
            raise ValueError("Transport reuse must bind the mixture property provider.")
        try:
            mode = LowMachPressureMode(pressure_mode)
        except ValueError as failure:
            raise ValueError("Unknown low-Mach thermodynamic pressure mode.") from failure
        conservation = float(conservation_tolerance)
        eos = float(eos_tolerance)
        iterations = int(maximum_temperature_iterations)
        if (
            not isfinite(conservation)
            or conservation <= 0.0
            or not isfinite(eos)
            or eos <= 0.0
            or iterations <= 0
        ):
            raise ValueError("Low-Mach tolerances or temperature iterations are invalid.")
        self.formulation = formulation
        self.mixture_transport = mixture_transport
        self.scalar_transport = scalar_transport
        self.projection = projection
        self.sdc = sdc
        self.transport_reuse = transport_reuse
        self.pressure_mode = mode
        self.species_field_names = species_names
        self.enthalpy_field_name = enthalpy_name
        self.conservation_tolerance = conservation
        self.eos_tolerance = eos
        self.maximum_temperature_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-spatial-low-mach-reacting-flow",
                "formulation": formulation.formulation_id,
                "mixture_transport": mixture_transport.transport_id,
                "scalar_transport": scalar_transport.prepared_id,
                "projection": projection.plan_id,
                "sdc": sdc.plan_id,
                "reuse": None if transport_reuse is None else transport_reuse.plan_id,
                "pressure_mode": mode.value,
                "conservation_tolerance": conservation,
                "eos_tolerance": eos,
                "maximum_temperature_iterations": iterations,
            }
        )

    @property
    def operators(self):
        return self.projection.operators

    def initialize(
        self,
        velocity: FaceVelocity,
        temperature: ArrayLike,
        mass_fractions: ArrayLike,
        thermodynamic_pressure: ArrayLike,
        /,
    ) -> LowMachReactingFlowState:
        velocity_ = self.operators.validate_velocity(velocity)
        temperature_ = jnp.asarray(temperature, dtype=self.operators.pressure_space.dtype)
        mass = jnp.asarray(mass_fractions, dtype=temperature_.dtype)
        pressure = jnp.asarray(thermodynamic_pressure, dtype=temperature_.dtype)
        shape = self.operators.discretization.cell_shape
        species_count = self.formulation.thermodynamics.schema.species_count
        if temperature_.shape != shape or mass.shape != shape + (species_count,):
            raise ValueError("Low-Mach temperature/mass fractions do not match cells.")
        if pressure.shape != ():
            raise ValueError("thermodynamic_pressure must be scalar.")
        thermo = self.formulation.pressure_state(temperature_, pressure, mass)
        species_density = thermo.mass_density[..., None] * mass
        enthalpy_density = thermo.molar_density * thermo.molar_enthalpy
        reuse = (
            None
            if self.transport_reuse is None
            else self.transport_reuse.initialize(
                temperature_, jnp.broadcast_to(pressure, shape)
            )
        )
        return LowMachReactingFlowState(
            velocity_,
            species_density,
            enthalpy_density,
            pressure,
            jnp.zeros(shape, dtype=temperature_.dtype),
            reuse,
            jnp.asarray(0.0, dtype=temperature_.dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def _recover(self, species_density: Array, enthalpy_density: Array, /):
        schema = self.formulation.thermodynamics.schema
        molar_mass = schema.molar_masses.astype(species_density.dtype)
        concentration = species_density / molar_mass
        molar_density = jnp.sum(concentration, axis=-1)
        mole_fraction = concentration / molar_density[..., None]
        lower = jnp.full_like(
            enthalpy_density,
            self.formulation.thermodynamics.thermodynamics.minimum_temperature,
        )
        upper = jnp.full_like(
            enthalpy_density,
            self.formulation.thermodynamics.thermodynamics.maximum_temperature,
        )

        def enthalpy(temperature):
            species = self.formulation.thermodynamics.thermodynamics.evaluate(temperature)
            return contract(
                "...s,...s->...",
                concentration,
                species.molar_enthalpy,
                backend="jax",
            )

        def iteration(_, bounds):
            low, high = bounds
            midpoint = 0.5 * (low + high)
            below = enthalpy(midpoint) < enthalpy_density
            return jnp.where(below, midpoint, low), jnp.where(below, high, midpoint)

        lower, upper = jax.lax.fori_loop(
            0, self.maximum_temperature_iterations, iteration, (lower, upper)
        )
        temperature = 0.5 * (lower + upper)
        thermal = self.formulation.thermodynamics.evaluate(
            temperature, molar_density, mole_fraction
        )
        residual = enthalpy(temperature) - enthalpy_density
        scale = jnp.maximum(jnp.abs(enthalpy_density), 1.0)
        successful = (
            thermal.evidence.successful
            & jnp.all(species_density >= 0.0, axis=-1)
            & jnp.isfinite(residual)
            & (jnp.abs(residual) <= self.eos_tolerance * scale)
        )
        return thermal, residual, successful

    def _cell_gradient(self, value: Array, /) -> Array:
        components = []
        for axis, face in enumerate(self.operators.gradient(value)):
            right = jnp.roll(face, -1, axis=axis)
            components.append(0.5 * (face + right))
        return jnp.stack(tuple(components), axis=-1)

    def _face_flux(self, vector: Array, /) -> FaceVelocity:
        output = []
        for axis in range(len(self.operators.discretization.cell_shape)):
            component = vector[..., axis]
            output.append(0.5 * (component + jnp.roll(component, 1, axis=axis)))
        return tuple(output)

    def _fields(self, species_density: Array, enthalpy_density: Array, /):
        output = {
            name: species_density[..., index]
            for index, name in enumerate(self.species_field_names)
        }
        output[self.enthalpy_field_name] = enthalpy_density
        return output

    def _rates(
        self,
        state: LowMachReactingFlowState,
        species_density: Array,
        enthalpy_density: Array,
        pressure: Array,
        pressure_rate: Array,
        /,
        *,
        args: Any,
    ) -> _LowMachRates:
        thermal, _, recovered = self._recover(species_density, enthalpy_density)
        density = jnp.sum(species_density, axis=-1)
        mass = species_density / density[..., None]
        temperature = thermal.temperature
        shape = self.operators.discretization.cell_shape
        scalar_fluxes = self.scalar_transport.evaluate(
            state.time,
            self._fields(species_density, enthalpy_density),
            state.velocity,
            args,
        )
        advection = {
            name: scalar_fluxes[name].rate
            for name in self.scalar_transport.layout.field_names
        }
        mass_gradient = jnp.stack(
            tuple(
                self._cell_gradient(mass[..., index]) for index in range(mass.shape[-1])
            ),
            axis=-2,
        )
        temperature_gradient = self._cell_gradient(temperature)
        reuse_candidate = (
            None
            if self.transport_reuse is None
            else self.transport_reuse.propose(
                state.transport_reuse,
                temperature,
                thermal.pressure,
            )
        )
        transport = self.mixture_transport.evaluate(
            temperature,
            thermal.pressure,
            density,
            mass,
            mass_gradient,
            temperature_gradient=temperature_gradient,
            property_evaluation=None
            if reuse_candidate is None
            else reuse_candidate.properties,
        )
        species_diffusion = jnp.stack(
            tuple(
                -self.operators.divergence(
                    self._face_flux(transport.species_mass_flux[..., index, :])
                )
                for index in range(mass.shape[-1])
            ),
            axis=-1,
        )
        enthalpy_diffusion = -self.operators.divergence(
            self._face_flux(transport.total_heat_flux)
        )
        concentrations = (
            species_density
            / self.formulation.mechanism.schema.molar_masses.astype(species_density.dtype)
        )
        chemistry = self.formulation.mechanism.evaluate(
            concentrations,
            temperature,
            jnp.broadcast_to(pressure, shape),
        )
        chemistry_mass_rate = (
            chemistry.species_amount_rate
            * self.formulation.mechanism.schema.molar_masses.astype(species_density.dtype)
        )
        species_rate = (
            jnp.stack(
                tuple(advection[name] for name in self.species_field_names), axis=-1
            )
            + species_diffusion
            + chemistry_mass_rate
        )
        enthalpy_rate = advection[self.enthalpy_field_name] + enthalpy_diffusion
        density_rate = jnp.sum(species_rate, axis=-1)
        mass_rate = (species_rate - mass * density_rate[..., None]) / density[..., None]
        species_thermo = self.formulation.thermodynamics.thermodynamics.evaluate(
            temperature
        )
        species_molar_masses = self.formulation.thermodynamics.schema.molar_masses.astype(
            species_density.dtype
        )
        species_specific_enthalpy = species_thermo.molar_enthalpy / species_molar_masses
        specific_enthalpy = enthalpy_density / density
        specific_enthalpy_rate = (
            enthalpy_rate - specific_enthalpy * density_rate
        ) / density
        cp = thermal.molar_heat_capacity_pressure / thermal.molar_mass
        temperature_rate = (
            specific_enthalpy_rate
            - contract(
                "...s,...s->...",
                species_specific_enthalpy,
                mass_rate,
                backend="jax",
            )
        ) / cp
        divergence = self.formulation.divergence_source(
            temperature,
            mass,
            temperature_rate,
            mass_rate,
            pressure,
            thermodynamic_pressure_rate=pressure_rate,
        )
        success_values = tuple(result.success for result in scalar_fluxes.values())
        successful = (
            recovered
            & transport.successful
            & chemistry.successful
            & divergence.successful
            & jnp.all(jnp.stack(success_values))
            & (
                jnp.asarray(True)
                if reuse_candidate is None
                else jnp.all(reuse_candidate.successful)
            )
        )
        return _LowMachRates(
            species_rate,
            enthalpy_rate,
            divergence.divergence_source,
            temperature,
            density,
            thermal.pressure,
            reuse_candidate,
            successful,
        )

    def advance(
        self,
        state: LowMachReactingFlowState,
        step_size: ArrayLike,
        /,
        *,
        thermodynamic_pressure_rate: ArrayLike = 0.0,
        external_species_rate: ArrayLike | None = None,
        external_enthalpy_rate: ArrayLike | None = None,
        args: Any = None,
    ) -> LowMachReactingStepResult:
        if (
            not isinstance(state, LowMachReactingFlowState)
            or state.plan_id != self.plan_id
        ):
            raise TypeError("state must belong to this LowMachReactingFlowPlan.")
        step = jnp.asarray(step_size, dtype=state.time.dtype)
        pressure_rate = jnp.asarray(thermodynamic_pressure_rate, dtype=state.time.dtype)
        if step.shape != () or pressure_rate.shape != ():
            raise ValueError("Low-Mach step and pressure rate must be scalar.")
        step_valid = jnp.isfinite(step) & (step > 0.0) & jnp.isfinite(pressure_rate)
        zero_species = jnp.zeros_like(state.species_density)
        zero_enthalpy = jnp.zeros_like(state.enthalpy_density)
        external_species = (
            zero_species
            if external_species_rate is None
            else jnp.asarray(external_species_rate, dtype=state.species_density.dtype)
        )
        external_enthalpy = (
            zero_enthalpy
            if external_enthalpy_rate is None
            else jnp.asarray(external_enthalpy_rate, dtype=state.enthalpy_density.dtype)
        )
        if (
            external_species.shape != zero_species.shape
            or external_enthalpy.shape != zero_enthalpy.shape
        ):
            raise ValueError("External low-Mach sources must match conservative fields.")
        if self.pressure_mode is LowMachPressureMode.CONSTANT:
            pressure_rate = eqx.error_if(
                pressure_rate,
                pressure_rate != 0.0,
                "Constant-pressure low-Mach mode requires zero pressure rate.",
            )
        first = self._rates(
            state,
            state.species_density,
            state.enthalpy_density,
            state.thermodynamic_pressure,
            pressure_rate,
            args=args,
        )
        first_species_rate = first.species_density + external_species
        first_enthalpy_rate = first.enthalpy_density + external_enthalpy
        species_candidate = state.species_density + step * first_species_rate
        enthalpy_candidate = state.enthalpy_density + step * first_enthalpy_rate
        sdc_residual = jnp.asarray(jnp.inf, dtype=state.time.dtype)
        final = first
        for _ in range(self.sdc.correction_sweeps):
            candidate_valid = (
                jnp.all(jnp.isfinite(species_candidate), axis=-1)
                & jnp.all(species_candidate > 0.0, axis=-1)
                & jnp.isfinite(enthalpy_candidate)
            )
            evaluation_species = jnp.where(
                candidate_valid[..., None],
                species_candidate,
                state.species_density,
            )
            evaluation_enthalpy = jnp.where(
                candidate_valid,
                enthalpy_candidate,
                state.enthalpy_density,
            )
            final = self._rates(
                state,
                evaluation_species,
                evaluation_enthalpy,
                state.thermodynamic_pressure + step * pressure_rate,
                pressure_rate,
                args=args,
            )
            corrected_species = state.species_density + 0.5 * step * (
                first_species_rate + final.species_density + external_species
            )
            corrected_enthalpy = state.enthalpy_density + 0.5 * step * (
                first_enthalpy_rate + final.enthalpy_density + external_enthalpy
            )
            scale = jnp.maximum(
                jnp.maximum(
                    jnp.max(jnp.abs(corrected_species)),
                    jnp.max(jnp.abs(corrected_enthalpy)),
                ),
                1.0,
            )
            sdc_residual = (
                jnp.maximum(
                    jnp.max(jnp.abs(corrected_species - species_candidate)),
                    jnp.max(jnp.abs(corrected_enthalpy - enthalpy_candidate)),
                )
                / scale
            )
            species_candidate, enthalpy_candidate = corrected_species, corrected_enthalpy
        candidate_valid = (
            jnp.all(jnp.isfinite(species_candidate), axis=-1)
            & jnp.all(species_candidate > 0.0, axis=-1)
            & jnp.isfinite(enthalpy_candidate)
        )
        evaluation_species = jnp.where(
            candidate_valid[..., None], species_candidate, state.species_density
        )
        evaluation_enthalpy = jnp.where(
            candidate_valid, enthalpy_candidate, state.enthalpy_density
        )
        thermal_candidate, _, recovered = self._recover(
            evaluation_species, evaluation_enthalpy
        )
        volumes = self.operators.discretization.cell_volumes.astype(state.time.dtype)
        if self.pressure_mode is LowMachPressureMode.CLOSED:
            pressure_candidate = state.thermodynamic_pressure
            final_pressure_rate = jnp.asarray(0.0, dtype=state.time.dtype)
            final = self._rates(
                state,
                evaluation_species,
                evaluation_enthalpy,
                pressure_candidate,
                final_pressure_rate,
                args=args,
            )
            mass = evaluation_species / jnp.sum(evaluation_species, axis=-1)[..., None]
            for _ in range(3):
                pressure_state = self.formulation.pressure_state(
                    final.temperature, pressure_candidate, mass
                )
                correction = jnp.sum(volumes * final.divergence_source) / jnp.sum(
                    volumes * pressure_state.isothermal_compressibility
                )
                final_pressure_rate = final_pressure_rate + correction
                pressure_candidate = (
                    state.thermodynamic_pressure + step * final_pressure_rate
                )
                final = self._rates(
                    state,
                    evaluation_species,
                    evaluation_enthalpy,
                    pressure_candidate,
                    final_pressure_rate,
                    args=args,
                )
            target_divergence = self.operators.compatibility_project(
                final.divergence_source
            )
        else:
            pressure_candidate = state.thermodynamic_pressure + step * pressure_rate
            final_pressure_rate = pressure_rate
            final = self._rates(
                state,
                evaluation_species,
                evaluation_enthalpy,
                pressure_candidate,
                final_pressure_rate,
                args=args,
            )
            target_divergence = final.divergence_source
        density = jnp.sum(evaluation_species, axis=-1)
        cell_inverse_density = 1.0 / density
        face_inverse_density = self.operators.interpolate_inverse_momentum(
            cell_inverse_density
        )
        momentum = tuple(
            velocity / inverse
            for velocity, inverse in zip(
                state.velocity, face_inverse_density, strict=True
            )
        )
        projected = self.projection.project(
            momentum,
            face_inverse_density,
            step,
            pressure=state.mechanical_pressure,
            target_divergence=target_divergence,
        )
        candidate_reuse = (
            state.transport_reuse
            if final.transport_reuse is None
            else final.transport_reuse.proposed_state
        )
        candidate = LowMachReactingFlowState(
            projected.velocity,
            species_candidate,
            enthalpy_candidate,
            pressure_candidate,
            projected.pressure,
            candidate_reuse,
            state.time + step,
            state.accepted_step + 1,
            self.plan_id,
        )
        initial_amount = (
            state.species_density
            / self.formulation.thermodynamics.schema.molar_masses.astype(
                state.species_density.dtype
            )
        )
        final_amount = (
            species_candidate
            / self.formulation.thermodynamics.schema.molar_masses.astype(
                state.species_density.dtype
            )
        )
        element_initial = jnp.sum(
            volumes[..., None]
            * self.formulation.thermodynamics.schema.element_amount(initial_amount),
            axis=tuple(range(volumes.ndim)),
        )
        element_final = jnp.sum(
            volumes[..., None]
            * self.formulation.thermodynamics.schema.element_amount(final_amount),
            axis=tuple(range(volumes.ndim)),
        )
        charge_initial = jnp.sum(
            volumes * self.formulation.thermodynamics.schema.charge_amount(initial_amount)
        )
        charge_final = jnp.sum(
            volumes * self.formulation.thermodynamics.schema.charge_amount(final_amount)
        )
        mass_initial = jnp.sum(volumes * jnp.sum(state.species_density, axis=-1))
        mass_final = jnp.sum(volumes * density)
        enthalpy_initial = jnp.sum(volumes * state.enthalpy_density)
        enthalpy_final = jnp.sum(volumes * enthalpy_candidate)
        element_defect = (
            element_final
            - element_initial
            - step
            * jnp.sum(
                volumes[..., None]
                * self.formulation.thermodynamics.schema.element_amount(
                    external_species
                    / self.formulation.thermodynamics.schema.molar_masses.astype(
                        external_species.dtype
                    )
                ),
                axis=tuple(range(volumes.ndim)),
            )
        )
        charge_defect = (
            charge_final
            - charge_initial
            - step
            * jnp.sum(
                volumes
                * self.formulation.thermodynamics.schema.charge_amount(
                    external_species
                    / self.formulation.thermodynamics.schema.molar_masses.astype(
                        external_species.dtype
                    )
                )
            )
        )
        mass_defect = (
            mass_final
            - mass_initial
            - step * jnp.sum(volumes * jnp.sum(external_species, axis=-1))
        )
        enthalpy_defect = (
            enthalpy_final
            - enthalpy_initial
            - step * jnp.sum(volumes * external_enthalpy)
        )
        eos_defect = thermal_candidate.pressure - pressure_candidate
        element_scale = jnp.maximum(jnp.max(jnp.abs(element_initial), initial=0.0), 1.0)
        charge_scale = jnp.maximum(jnp.abs(charge_initial), 1.0)
        mass_scale = jnp.maximum(jnp.abs(mass_initial), 1.0)
        enthalpy_scale = jnp.maximum(jnp.abs(enthalpy_initial), 1.0)
        conservative = (
            (
                jnp.max(jnp.abs(element_defect), initial=0.0)
                <= self.conservation_tolerance * element_scale
            )
            & (jnp.abs(charge_defect) <= self.conservation_tolerance * charge_scale)
            & (jnp.abs(mass_defect) <= self.conservation_tolerance * mass_scale)
            & (jnp.abs(enthalpy_defect) <= self.conservation_tolerance * enthalpy_scale)
        )
        minimum_species = jnp.min(species_candidate)
        finite = (
            jnp.all(jnp.isfinite(species_candidate))
            & jnp.all(jnp.isfinite(enthalpy_candidate))
            & jnp.all(jnp.isfinite(eos_defect))
            & jnp.isfinite(sdc_residual)
        )
        admissible = (
            jnp.all(recovered)
            & (minimum_species >= 0.0)
            & jnp.all(
                jnp.abs(eos_defect)
                <= self.eos_tolerance * jnp.maximum(jnp.abs(pressure_candidate), 1.0)
            )
        )
        successful = (
            step_valid
            & jnp.all(first.successful)
            & jnp.all(final.successful)
            & projected.successful
            & finite
            & admissible
            & conservative
            & (sdc_residual <= self.sdc.nonlinear_tolerance)
        )
        accepted_reuse = state.transport_reuse
        if self.transport_reuse is not None:
            accepted_reuse = self.transport_reuse.commit(
                state.transport_reuse, final.transport_reuse, successful
            )
        accepted_candidate = eqx.tree_at(
            lambda value: value.transport_reuse, candidate, accepted_reuse
        )
        accepted = jax.tree.map(
            lambda proposed, prior: jnp.where(successful, proposed, prior),
            accepted_candidate,
            state,
        )
        diagnostics = LowMachReactingStepDiagnostics(
            element_defect,
            charge_defect,
            mass_defect,
            enthalpy_defect,
            eos_defect,
            projected.divergence_defect,
            sdc_residual,
            minimum_species,
            finite,
            admissible,
            conservative,
            successful,
            self.plan_id,
        )
        return LowMachReactingStepResult(
            candidate,
            accepted,
            projected,
            diagnostics,
            successful,
            self.plan_id,
        )


__all__ = [
    "LowMachPressureMode",
    "LowMachReactingFlowPlan",
    "LowMachReactingFlowState",
    "LowMachReactingSDCPlan",
    "LowMachReactingStepDiagnostics",
    "LowMachReactingStepResult",
]
