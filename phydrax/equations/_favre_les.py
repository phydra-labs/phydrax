#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._les_closures import (
    AlgebraicLESInputs,
    LESFilterScale,
    PreparedAlgebraicLESModel,
)


_FavreIsotropicTracePolicy = Literal["neglected", "provided-sgs-kinetic-energy"]


class FavreLESFieldContract(StrictModule, NonTrainableState):
    """Exact field ordering and physical-unit identity for Favre transport.

    The first runtime route intentionally accepts only the canonical SI gas fields
    used by :mod:`phydrax.equations._gas_dynamics`: total and species densities in
    kg/m³, absolute temperature in kelvin, and dimensionless mass fractions.  A
    unit conversion must therefore happen before this contract is constructed;
    accepting a merely named, unscaled alternative would make the SGS transport
    coefficients physically ambiguous.
    """

    schema_id: str = eqx.field(static=True)
    species_names: tuple[str, ...] = eqx.field(static=True)
    density_unit: str = eqx.field(static=True)
    temperature_unit: str = eqx.field(static=True)
    species_density_unit: str = eqx.field(static=True)
    species_fraction_unit: str = eqx.field(static=True)
    velocity_unit: str = eqx.field(static=True)
    filter_scale_unit: str = eqx.field(static=True)
    specific_enthalpy_unit: str = eqx.field(static=True)
    specific_heat_capacity_unit: str = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema_id: str,
        species_names: tuple[str, ...],
        /,
        *,
        density_unit: str = "kg/m^3",
        temperature_unit: str = "K",
        species_density_unit: str = "kg/m^3",
        species_fraction_unit: str = "kg/kg",
        velocity_unit: str = "m/s",
        filter_scale_unit: str = "m",
        specific_enthalpy_unit: str = "J/kg",
        specific_heat_capacity_unit: str = "J/(kg*K)",
    ):
        if not isinstance(schema_id, str) or not schema_id.strip():
            raise ValueError("Favre LES schema_id must be a non-empty string.")
        if (
            not isinstance(species_names, tuple)
            or not species_names
            or any(
                not isinstance(name, str) or not name.strip() for name in species_names
            )
        ):
            raise TypeError("Favre LES species_names must be non-empty strings.")
        names = tuple(name.strip() for name in species_names)
        if len(set(names)) != len(names):
            raise ValueError("Favre LES species names must be unique.")
        supplied_units = (
            density_unit,
            temperature_unit,
            species_density_unit,
            species_fraction_unit,
            velocity_unit,
            filter_scale_unit,
            specific_enthalpy_unit,
            specific_heat_capacity_unit,
        )
        if any(not isinstance(unit, str) for unit in supplied_units):
            raise TypeError("Favre LES units must be strings.")
        canonical_units = (
            "kg/m^3",
            "K",
            "kg/m^3",
            "kg/kg",
            "m/s",
            "m",
            "J/kg",
            "J/(kg*K)",
        )
        if supplied_units != canonical_units:
            raise ValueError(
                "Favre LES requires exact canonical units (SI) for density, absolute "
                "temperature, species density/mass fraction, velocity, filter "
                "scale, specific enthalpy, and specific heat capacity."
            )
        schema = schema_id.strip()
        self.schema_id = schema
        self.species_names = names
        self.density_unit = canonical_units[0]
        self.temperature_unit = canonical_units[1]
        self.species_density_unit = canonical_units[2]
        self.species_fraction_unit = canonical_units[3]
        self.velocity_unit = canonical_units[4]
        self.filter_scale_unit = canonical_units[5]
        self.specific_enthalpy_unit = canonical_units[6]
        self.specific_heat_capacity_unit = canonical_units[7]
        self.contract_id = canonical_fingerprint(
            {
                "kind": "favre-les-field-contract",
                "schema": schema,
                "species_names": names,
                "density_unit": canonical_units[0],
                "temperature_unit": canonical_units[1],
                "species_density_unit": canonical_units[2],
                "species_fraction_unit": canonical_units[3],
                "velocity_unit": canonical_units[4],
                "filter_scale_unit": canonical_units[5],
                "specific_enthalpy_unit": canonical_units[6],
                "specific_heat_capacity_unit": canonical_units[7],
            }
        )


class FavreLESInputEvidence(StrictModule):
    """Pointwise admissibility evidence for one Favre closure evaluation."""

    finite: Array
    density_positive: Array
    temperature_positive: Array
    heat_capacity_positive: Array
    species_nonnegative: Array
    sgs_kinetic_energy_nonnegative: Array
    mass_fraction_closed: Array
    successful: Array
    contract_id: str = eqx.field(static=True)


class FavreLESResultEvidence(StrictModule):
    """Finite-output and configured eddy-viscosity-bound evidence."""

    finite: Array
    viscosity_nonnegative: Array
    viscosity_within_bound: Array
    dissipation_nonnegative: Array
    successful: Array
    closure_id: str = eqx.field(static=True)


class FavreLESInputs(StrictModule):
    """Resolved variable-density fields needed by an algebraic Favre closure."""

    density: Array
    temperature: Array
    favre_velocity: Array
    favre_velocity_gradient: Array
    temperature_gradient: Array
    mass_fractions: Array
    mass_fraction_gradient: Array
    specific_sgs_kinetic_energy_gradient: Array | None
    specific_heat_capacity_pressure: Array
    partial_specific_enthalpies: Array
    specific_sgs_kinetic_energy: Array | None
    fields: FavreLESFieldContract

    def __init__(
        self,
        density: ArrayLike,
        temperature: ArrayLike,
        favre_velocity: ArrayLike,
        favre_velocity_gradient: ArrayLike,
        temperature_gradient: ArrayLike,
        mass_fractions: ArrayLike,
        mass_fraction_gradient: ArrayLike,
        specific_heat_capacity_pressure: ArrayLike,
        partial_specific_enthalpies: ArrayLike,
        fields: FavreLESFieldContract,
        /,
        *,
        specific_sgs_kinetic_energy: ArrayLike | None = None,
        specific_sgs_kinetic_energy_gradient: ArrayLike | None = None,
    ):
        if not isinstance(fields, FavreLESFieldContract):
            raise TypeError("fields must be a FavreLESFieldContract.")
        density_ = jnp.asarray(density)
        if not jnp.issubdtype(density_.dtype, jnp.inexact):
            density_ = density_.astype(jnp.result_type(density_, float))
        temperature_ = jnp.asarray(temperature, dtype=density_.dtype)
        velocity = jnp.asarray(favre_velocity, dtype=density_.dtype)
        velocity_gradient = jnp.asarray(favre_velocity_gradient, dtype=density_.dtype)
        thermal_gradient = jnp.asarray(temperature_gradient, dtype=density_.dtype)
        fractions = jnp.asarray(mass_fractions, dtype=density_.dtype)
        fraction_gradient = jnp.asarray(mass_fraction_gradient, dtype=density_.dtype)
        heat_capacity = jnp.asarray(specific_heat_capacity_pressure, dtype=density_.dtype)
        enthalpies = jnp.asarray(partial_specific_enthalpies, dtype=density_.dtype)
        batch_shape = density_.shape
        species_count = len(fields.species_names)
        expected = (
            temperature_.shape == batch_shape,
            velocity.shape == batch_shape + (3,),
            velocity_gradient.shape == batch_shape + (3, 3),
            thermal_gradient.shape == batch_shape + (3,),
            fractions.shape == batch_shape + (species_count,),
            fraction_gradient.shape == batch_shape + (species_count, 3),
            heat_capacity.shape == batch_shape,
            enthalpies.shape == batch_shape + (species_count,),
        )
        if not all(expected):
            raise ValueError(
                "Favre LES fields must share leading dimensions and append exact "
                "three-dimensional velocity or named-species axes."
            )
        kinetic_energy = None
        kinetic_energy_gradient = None
        if specific_sgs_kinetic_energy is not None:
            kinetic_energy = jnp.asarray(
                specific_sgs_kinetic_energy, dtype=density_.dtype
            )
            if kinetic_energy.shape != batch_shape:
                raise ValueError(
                    "specific_sgs_kinetic_energy must match the density shape."
                )
        if specific_sgs_kinetic_energy_gradient is not None:
            kinetic_energy_gradient = jnp.asarray(
                specific_sgs_kinetic_energy_gradient, dtype=density_.dtype
            )
            if kinetic_energy_gradient.shape != batch_shape + (3,):
                raise ValueError(
                    "specific_sgs_kinetic_energy_gradient must append one "
                    "three-dimensional derivative axis to the density shape."
                )
        self.density = density_
        self.temperature = temperature_
        self.favre_velocity = velocity
        self.favre_velocity_gradient = velocity_gradient
        self.temperature_gradient = thermal_gradient
        self.mass_fractions = fractions
        self.mass_fraction_gradient = fraction_gradient
        self.specific_heat_capacity_pressure = heat_capacity
        self.partial_specific_enthalpies = enthalpies
        self.specific_sgs_kinetic_energy = kinetic_energy
        self.specific_sgs_kinetic_energy_gradient = kinetic_energy_gradient
        self.fields = fields


class FavreLESResult(StrictModule):
    """Favre SGS transport and the complete transported-energy ledger.

    ``sgs_stress`` and ``sgs_species_flux`` use unresolved-covariance signs.
    Their conservative diffusive images therefore carry a minus sign.  With
    transported SGS kinetic energy, its diffusive flux is added to both the SGS
    and total-energy equations.  Production and dissipation occur only in the
    SGS-energy equation: the conserved total-energy source remains identically
    zero, so their difference is an internal exchange with resolved energy.
    """

    kinematic_eddy_viscosity: Array
    dynamic_eddy_viscosity: Array
    density_weighted_deviatoric_sgs_stress: Array
    isotropic_sgs_stress: Array
    sgs_stress: Array
    specific_sgs_kinetic_energy: Array
    sgs_kinetic_energy_density: Array
    sgs_heat_flux: Array
    sgs_species_flux: Array
    sgs_species_enthalpy_flux: Array
    sgs_enthalpy_flux: Array
    deviatoric_energy_transfer: Array
    isotropic_energy_transfer: Array
    sgs_kinetic_energy_production: Array
    sgs_kinetic_energy_dissipation: Array
    sgs_kinetic_energy_source: Array
    sgs_kinetic_energy_diffusion_flux: Array
    conservative_species_flux: Array
    conservative_momentum_flux: Array
    deviatoric_stress_work_flux: Array
    isotropic_stress_work_flux: Array
    stress_work_flux: Array
    conservative_resolved_energy_flux: Array
    conservative_total_energy_flux: Array
    input_evidence: FavreLESInputEvidence
    evidence: FavreLESResultEvidence
    fields: FavreLESFieldContract

    def species_flux(self, name: str, /) -> Array:
        """Return one conventional SGS species flux by its schema name."""
        if not isinstance(name, str) or name not in self.fields.species_names:
            raise KeyError(f"Unknown Favre LES species {name!r}.")
        return self.sgs_species_flux[..., self.fields.species_names.index(name), :]

    def source_positivity_timestep(self) -> Array:
        """Return the exact forward-Euler bound for the local SGS-energy sink."""
        sink = jnp.maximum(-self.sgs_kinetic_energy_source, 0.0)
        active = sink > 0.0
        safe_sink = jnp.where(active, sink, jnp.ones_like(sink))
        return jnp.where(
            active,
            self.sgs_kinetic_energy_density / safe_sink,
            jnp.full_like(sink, jnp.inf),
        )


class PreparedFavreLESModel(StrictModule, NonTrainableState):
    """Prepared three-dimensional algebraic Favre effective-transport closure.

    This closure is physical SGS transport, never a shock sensor, Riemann
    dissipation, limiter, or artificial viscosity.  Its spatial filter and
    coefficient provenance are inherited only from the explicitly supplied
    prepared algebraic model; no periodic, MAC, or unit-density support is
    implied by this type.
    """

    algebraic_model: PreparedAlgebraicLESModel
    filter_scale: LESFilterScale
    fields: FavreLESFieldContract
    turbulent_prandtl_number: float = eqx.field(static=True)
    species_turbulent_schmidt_numbers: tuple[tuple[str, float], ...] = eqx.field(
        static=True
    )
    isotropic_trace_policy: _FavreIsotropicTracePolicy = eqx.field(static=True)
    sgs_kinetic_energy_dissipation_coefficient: float = eqx.field(static=True)
    sgs_kinetic_energy_turbulent_schmidt_number: float = eqx.field(static=True)
    kinematic_viscosity_upper_bound: float = eqx.field(static=True)
    formulation: str = eqx.field(static=True)
    transport_role: str = eqx.field(static=True)
    numerical_stabilization_included: bool = eqx.field(static=True)
    closure_id: str = eqx.field(static=True)

    def __init__(
        self,
        algebraic_model: PreparedAlgebraicLESModel,
        filter_scale: LESFilterScale,
        fields: FavreLESFieldContract,
        turbulent_prandtl_number: float,
        species_turbulent_schmidt_numbers: tuple[tuple[str, float], ...],
        kinematic_viscosity_upper_bound: float,
        /,
        *,
        isotropic_trace_policy: _FavreIsotropicTracePolicy = "neglected",
        sgs_kinetic_energy_dissipation_coefficient: float = 1.05,
        sgs_kinetic_energy_turbulent_schmidt_number: float = 1.0,
    ):
        if not isinstance(algebraic_model, PreparedAlgebraicLESModel):
            raise TypeError("algebraic_model must be a PreparedAlgebraicLESModel.")
        if not isinstance(filter_scale, LESFilterScale):
            raise TypeError("filter_scale must be a LESFilterScale.")
        if not isinstance(fields, FavreLESFieldContract):
            raise TypeError("fields must be a FavreLESFieldContract.")
        if isinstance(filter_scale.directional_widths, jax.core.Tracer):
            raise TypeError("Prepared Favre LES filter widths must be concrete.")
        widths = np.asarray(filter_scale.directional_widths)
        if np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
            raise ValueError(
                "Prepared Favre LES filter widths must be positive and finite."
            )
        prandtl = float(turbulent_prandtl_number)
        upper_bound = float(kinematic_viscosity_upper_bound)
        dissipation_coefficient = float(sgs_kinetic_energy_dissipation_coefficient)
        kinetic_schmidt = float(sgs_kinetic_energy_turbulent_schmidt_number)
        if not np.isfinite(prandtl) or prandtl <= 0.0:
            raise ValueError(
                "Favre turbulent Prandtl number must be finite and positive."
            )
        if not np.isfinite(upper_bound) or upper_bound < 0.0:
            raise ValueError(
                "Favre kinematic viscosity upper bound must be finite and nonnegative."
            )
        if not np.isfinite(dissipation_coefficient) or dissipation_coefficient < 0.0:
            raise ValueError(
                "Favre SGS kinetic-energy dissipation coefficient must be finite "
                "and nonnegative."
            )
        if not np.isfinite(kinetic_schmidt) or kinetic_schmidt <= 0.0:
            raise ValueError(
                "Favre SGS kinetic-energy turbulent Schmidt number must be finite "
                "and positive."
            )
        if isotropic_trace_policy not in (
            "neglected",
            "provided-sgs-kinetic-energy",
        ):
            raise ValueError("Unsupported Favre isotropic SGS trace policy.")
        if not isinstance(species_turbulent_schmidt_numbers, tuple):
            raise TypeError(
                "species_turbulent_schmidt_numbers must be a tuple of named values."
            )
        schmidt_entries = tuple(species_turbulent_schmidt_numbers)
        if len(schmidt_entries) != len(fields.species_names) or any(
            not isinstance(entry, tuple)
            or len(entry) != 2
            or not isinstance(entry[0], str)
            for entry in schmidt_entries
        ):
            raise ValueError(
                "Favre turbulent Schmidt numbers require one named entry per species."
            )
        schmidt_names = tuple(entry[0] for entry in schmidt_entries)
        if schmidt_names != fields.species_names:
            raise ValueError(
                "Favre turbulent Schmidt entries must exactly follow species_names."
            )
        schmidt = tuple(float(entry[1]) for entry in schmidt_entries)
        if any(not np.isfinite(value) or value <= 0.0 for value in schmidt):
            raise ValueError(
                "Favre turbulent Schmidt numbers must be finite and positive."
            )
        entries = tuple(zip(schmidt_names, schmidt, strict=True))
        self.algebraic_model = algebraic_model
        self.filter_scale = filter_scale
        self.fields = fields
        self.turbulent_prandtl_number = prandtl
        self.species_turbulent_schmidt_numbers = entries
        self.isotropic_trace_policy = isotropic_trace_policy
        self.sgs_kinetic_energy_dissipation_coefficient = dissipation_coefficient
        self.sgs_kinetic_energy_turbulent_schmidt_number = kinetic_schmidt
        self.kinematic_viscosity_upper_bound = upper_bound
        self.formulation = "variable-density-favre-filtered"
        self.transport_role = "physical-subgrid-transport"
        self.numerical_stabilization_included = False
        self.closure_id = canonical_fingerprint(
            {
                "kind": "prepared-favre-les-effective-transport",
                "algebraic_model": algebraic_model.prepared_id,
                "filter": algebraic_model.provenance.resolved_filter.filter_id,
                "filter_scale": array_tree_fingerprint(widths),
                "fields": fields.contract_id,
                "turbulent_prandtl_number": prandtl,
                "species_turbulent_schmidt_numbers": entries,
                "isotropic_trace_policy": isotropic_trace_policy,
                "sgs_kinetic_energy_dissipation_coefficient": dissipation_coefficient,
                "sgs_kinetic_energy_turbulent_schmidt_number": kinetic_schmidt,
                "kinematic_viscosity_upper_bound": upper_bound,
                "transport_role": "physical-subgrid-transport",
                "numerical_stabilization_included": False,
            }
        )

    @property
    def provenance(self):
        """Return the complete resolved-filter and coefficient provenance."""
        return self.algebraic_model.provenance

    def validate_compressible_transport_binding(
        self,
        schema_id: str,
        species_names: tuple[str, ...],
        dimension: int,
        /,
    ) -> None:
        """Fail closed when the current gas transport path cannot carry the model."""
        if int(dimension) != 3:
            raise ValueError(
                "The initial Favre LES gas transport integration is three-dimensional."
            )
        if (
            schema_id != self.fields.schema_id
            or species_names != self.fields.species_names
        ):
            raise ValueError(
                "Favre LES field contract does not match the gas species schema."
            )

    def maximum_kinematic_diffusivity(self) -> float:
        """Return the configured conservative transport bound for timestep control."""
        species_minimum = min(
            value for _, value in self.species_turbulent_schmidt_numbers
        )
        diffusivities = [
            self.kinematic_viscosity_upper_bound,
            self.kinematic_viscosity_upper_bound / self.turbulent_prandtl_number,
            self.kinematic_viscosity_upper_bound / species_minimum,
        ]
        if self.isotropic_trace_policy == "provided-sgs-kinetic-energy":
            diffusivities.append(
                self.kinematic_viscosity_upper_bound
                / self.sgs_kinetic_energy_turbulent_schmidt_number
            )
        return max(diffusivities)

    def evaluate(self, inputs: FavreLESInputs, /) -> FavreLESResult:
        """Evaluate SGS transport, energy exchange, and conservative fluxes."""
        if not isinstance(inputs, FavreLESInputs):
            raise TypeError("inputs must be FavreLESInputs.")
        if inputs.fields.contract_id != self.fields.contract_id:
            raise ValueError("Favre LES inputs use a different field contract.")
        widths = self.filter_scale.directional_widths
        if widths.shape != (3,) and widths.shape != inputs.density.shape + (3,):
            raise ValueError(
                "Favre LES filter widths must be global or match the input leading shape."
            )
        if self.isotropic_trace_policy == "neglected":
            if (
                inputs.specific_sgs_kinetic_energy is not None
                or inputs.specific_sgs_kinetic_energy_gradient is not None
            ):
                raise ValueError(
                    "The neglected isotropic trace policy forbids SGS kinetic "
                    "energy and its gradient."
                )
            kinetic_energy = jnp.zeros_like(inputs.density)
            kinetic_energy_gradient = jnp.zeros(
                inputs.density.shape + (3,), dtype=inputs.density.dtype
            )
        else:
            if (
                inputs.specific_sgs_kinetic_energy is None
                or inputs.specific_sgs_kinetic_energy_gradient is None
            ):
                raise ValueError(
                    "The provided isotropic trace policy requires SGS kinetic "
                    "energy and its gradient."
                )
            kinetic_energy = inputs.specific_sgs_kinetic_energy
            kinetic_energy_gradient = inputs.specific_sgs_kinetic_energy_gradient

        fractions = inputs.mass_fractions
        fraction_residual = jnp.abs(jnp.sum(fractions, axis=-1) - 1.0)
        closure_tolerance = (
            32.0
            * jnp.finfo(inputs.density.dtype).eps
            * max(len(self.fields.species_names), 1)
        )
        finite = (
            jnp.isfinite(inputs.density)
            & jnp.isfinite(inputs.temperature)
            & jnp.all(jnp.isfinite(inputs.favre_velocity), axis=-1)
            & jnp.all(jnp.isfinite(inputs.favre_velocity_gradient), axis=(-2, -1))
            & jnp.all(jnp.isfinite(inputs.temperature_gradient), axis=-1)
            & jnp.all(jnp.isfinite(fractions), axis=-1)
            & jnp.all(jnp.isfinite(inputs.mass_fraction_gradient), axis=(-2, -1))
            & jnp.isfinite(inputs.specific_heat_capacity_pressure)
            & jnp.all(jnp.isfinite(inputs.partial_specific_enthalpies), axis=-1)
            & jnp.isfinite(kinetic_energy)
            & jnp.all(jnp.isfinite(kinetic_energy_gradient), axis=-1)
        )
        density_positive = inputs.density > 0.0
        temperature_positive = inputs.temperature > 0.0
        heat_capacity_positive = inputs.specific_heat_capacity_pressure > 0.0
        species_nonnegative = jnp.all(fractions >= 0.0, axis=-1)
        kinetic_energy_nonnegative = kinetic_energy >= 0.0
        mass_fraction_closed = fraction_residual <= closure_tolerance
        input_successful = (
            finite
            & density_positive
            & temperature_positive
            & heat_capacity_positive
            & species_nonnegative
            & kinetic_energy_nonnegative
            & mass_fraction_closed
        )
        density = eqx.error_if(
            inputs.density,
            jnp.any(~input_successful),
            "Favre LES requires finite positive density, temperature and heat "
            "capacity, finite nonnegative closed species mass fractions, finite "
            "gradients and nonnegative SGS kinetic energy.",
        )
        checked_kinetic_energy = eqx.error_if(
            kinetic_energy,
            jnp.any(~input_successful),
            "Favre LES requires nonnegative finite SGS kinetic energy.",
        )
        input_evidence = FavreLESInputEvidence(
            finite=finite,
            density_positive=density_positive,
            temperature_positive=temperature_positive,
            heat_capacity_positive=heat_capacity_positive,
            species_nonnegative=species_nonnegative,
            sgs_kinetic_energy_nonnegative=kinetic_energy_nonnegative,
            mass_fraction_closed=mass_fraction_closed,
            successful=input_successful,
            contract_id=self.fields.contract_id,
        )

        algebraic = self.algebraic_model.evaluate(
            AlgebraicLESInputs(inputs.favre_velocity_gradient, self.filter_scale)
        )
        kinematic_viscosity = algebraic.kinematic_viscosity
        viscosity_nonnegative = kinematic_viscosity >= 0.0
        viscosity_within_bound = (
            kinematic_viscosity <= self.kinematic_viscosity_upper_bound
        )
        checked_viscosity = eqx.error_if(
            kinematic_viscosity,
            jnp.any(
                ~jnp.isfinite(kinematic_viscosity)
                | ~viscosity_nonnegative
                | ~viscosity_within_bound
            ),
            "Favre LES eddy viscosity is non-finite, negative, or exceeds its "
            "configured timestep-control bound.",
        )
        dynamic_viscosity = density * checked_viscosity
        deviatoric_stress = (
            density[..., None, None] * algebraic.specific_deviatoric_stress
        )
        identity = jnp.eye(3, dtype=density.dtype)
        isotropic_stress = (
            (2.0 / 3.0)
            * density[..., None, None]
            * checked_kinetic_energy[..., None, None]
            * identity
        )
        stress = deviatoric_stress + isotropic_stress

        conductivity = (
            dynamic_viscosity
            * inputs.specific_heat_capacity_pressure
            / self.turbulent_prandtl_number
        )
        heat_flux = -conductivity[..., None] * inputs.temperature_gradient
        schmidt = jnp.asarray(
            tuple(value for _, value in self.species_turbulent_schmidt_numbers),
            dtype=density.dtype,
        )
        raw_species_flux = (
            -density[..., None, None]
            * checked_viscosity[..., None, None]
            * inputs.mass_fraction_gradient
            / schmidt[:, None]
        )
        species_flux = (
            raw_species_flux
            - fractions[..., :, None] * jnp.sum(raw_species_flux, axis=-2)[..., None, :]
        )
        species_enthalpy_flux = ein.contract(
            "...s,...sd->...d",
            inputs.partial_specific_enthalpies,
            species_flux,
            backend="jax",
        )
        enthalpy_flux = heat_flux + species_enthalpy_flux
        kinetic_energy_density = density * checked_kinetic_energy
        kinetic_energy_diffusion_flux = (
            dynamic_viscosity[..., None]
            * kinetic_energy_gradient
            / self.sgs_kinetic_energy_turbulent_schmidt_number
        )
        deviatoric_transfer = density * algebraic.energy_transfer
        isotropic_transfer = -ein.contract(
            "...ij,...ij->...",
            isotropic_stress,
            inputs.favre_velocity_gradient,
            backend="jax",
        )
        production = deviatoric_transfer + isotropic_transfer
        dissipation = (
            density
            * self.sgs_kinetic_energy_dissipation_coefficient
            * checked_kinetic_energy
            * jnp.sqrt(checked_kinetic_energy)
            / self.filter_scale.equivalent_width
        )
        source = production - dissipation
        dissipation_nonnegative = dissipation >= 0.0

        conservative_species_flux = -species_flux
        conservative_momentum_flux = -stress
        deviatoric_stress_work = -ein.contract(
            "...i,...ij->...j",
            inputs.favre_velocity,
            deviatoric_stress,
            backend="jax",
        )
        isotropic_stress_work = -ein.contract(
            "...i,...ij->...j",
            inputs.favre_velocity,
            isotropic_stress,
            backend="jax",
        )
        stress_work = deviatoric_stress_work + isotropic_stress_work
        conservative_resolved_energy_flux = stress_work - enthalpy_flux
        conservative_total_energy_flux = (
            conservative_resolved_energy_flux + kinetic_energy_diffusion_flux
        )
        output_finite = (
            jnp.isfinite(checked_viscosity)
            & jnp.isfinite(dynamic_viscosity)
            & jnp.all(jnp.isfinite(stress), axis=(-2, -1))
            & jnp.all(jnp.isfinite(heat_flux), axis=-1)
            & jnp.all(jnp.isfinite(species_flux), axis=(-2, -1))
            & jnp.isfinite(production)
            & jnp.isfinite(dissipation)
            & jnp.isfinite(source)
            & jnp.all(jnp.isfinite(kinetic_energy_diffusion_flux), axis=-1)
            & jnp.all(jnp.isfinite(conservative_total_energy_flux), axis=-1)
        )
        output_successful = (
            input_successful
            & output_finite
            & viscosity_nonnegative
            & viscosity_within_bound
            & dissipation_nonnegative
        )
        checked_energy_flux = eqx.error_if(
            conservative_total_energy_flux,
            jnp.any(~output_successful),
            "Favre LES produced non-finite or unsupported transport output.",
        )
        evidence = FavreLESResultEvidence(
            finite=output_finite,
            viscosity_nonnegative=viscosity_nonnegative,
            viscosity_within_bound=viscosity_within_bound,
            dissipation_nonnegative=dissipation_nonnegative,
            successful=output_successful,
            closure_id=self.closure_id,
        )
        return FavreLESResult(
            kinematic_eddy_viscosity=checked_viscosity,
            dynamic_eddy_viscosity=dynamic_viscosity,
            density_weighted_deviatoric_sgs_stress=deviatoric_stress,
            isotropic_sgs_stress=isotropic_stress,
            sgs_stress=stress,
            specific_sgs_kinetic_energy=checked_kinetic_energy,
            sgs_kinetic_energy_density=kinetic_energy_density,
            sgs_heat_flux=heat_flux,
            sgs_species_flux=species_flux,
            sgs_species_enthalpy_flux=species_enthalpy_flux,
            sgs_enthalpy_flux=enthalpy_flux,
            deviatoric_energy_transfer=deviatoric_transfer,
            isotropic_energy_transfer=isotropic_transfer,
            sgs_kinetic_energy_production=production,
            sgs_kinetic_energy_dissipation=dissipation,
            sgs_kinetic_energy_source=source,
            sgs_kinetic_energy_diffusion_flux=kinetic_energy_diffusion_flux,
            conservative_species_flux=conservative_species_flux,
            conservative_momentum_flux=conservative_momentum_flux,
            deviatoric_stress_work_flux=deviatoric_stress_work,
            isotropic_stress_work_flux=isotropic_stress_work,
            stress_work_flux=stress_work,
            conservative_resolved_energy_flux=conservative_resolved_energy_flux,
            conservative_total_energy_flux=checked_energy_flux,
            input_evidence=input_evidence,
            evidence=evidence,
            fields=self.fields,
        )


__all__ = [
    "FavreLESFieldContract",
    "FavreLESInputEvidence",
    "FavreLESInputs",
    "FavreLESResult",
    "FavreLESResultEvidence",
    "PreparedFavreLESModel",
]
