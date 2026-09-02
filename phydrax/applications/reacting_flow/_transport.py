#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg._dense_inverse import dense_inverse
from ._thermodynamics import ReactingGasModel


class ReactiveTransportEvaluation(StrictModule):
    dynamic_viscosity: Array
    thermal_conductivity: Array
    binary_diffusion_coefficients: Array
    mixture_diffusion_coefficients: Array
    diffusion_velocities: Array
    species_mass_flux: Array
    conductive_heat_flux: Array
    species_enthalpy_flux: Array
    total_heat_flux: Array
    net_mass_flux: Array
    composition_residual: Array
    successful: Array
    transport_id: str = eqx.field(static=True)


class StefanMaxwellEvidence(StrictModule):
    system_matrix: Array
    right_hand_side: Array
    velocity_residual: Array
    mass_constraint_residual: Array
    condition_estimate: Array
    successful: Array
    transport_id: str = eqx.field(static=True)


class StefanMaxwellTransportEvaluation(StrictModule):
    dynamic_viscosity: Array
    thermal_conductivity: Array
    binary_diffusion_coefficients: Array
    mixture_diffusion_coefficients: Array
    diffusion_velocities: Array
    species_mass_flux: Array
    conductive_heat_flux: Array
    species_enthalpy_flux: Array
    total_heat_flux: Array
    net_mass_flux: Array
    composition_residual: Array
    successful: Array
    transport_id: str = eqx.field(static=True)
    evidence: StefanMaxwellEvidence


def _validated_species_properties(values, species_count: int, name: str, /) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if (
        array.shape != (species_count,)
        or np.any(~np.isfinite(array))
        or np.any(array <= 0.0)
    ):
        raise ValueError(f"{name} must contain one finite positive value per species.")
    return array


def _validated_binary_diffusion(values, species_count: int, /) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (species_count, species_count):
        raise ValueError("binary_diffusion_coefficients has an invalid shape.")
    off_diagonal = ~np.eye(species_count, dtype=bool)
    if (
        np.any(~np.isfinite(matrix[off_diagonal]))
        or np.any(matrix[off_diagonal] <= 0.0)
        or not np.allclose(
            matrix[off_diagonal],
            matrix.T[off_diagonal],
            rtol=1.0e-12,
            atol=0.0,
        )
    ):
        raise ValueError(
            "Binary diffusion coefficients must be finite, positive off-diagonal, "
            "and symmetric."
        )
    matrix = matrix.copy()
    np.fill_diagonal(matrix, np.inf)
    return matrix


def _wilke_mixture(
    mole_fractions: Array, properties: Array, molar_masses: Array, /
) -> Array:
    ratio_property = properties[..., :, None] / properties[..., None, :]
    ratio_mass = molar_masses[None, :] / molar_masses[:, None]
    phi = (1.0 + jnp.sqrt(ratio_property) * ratio_mass**0.25) ** 2 / jnp.sqrt(
        8.0 * (1.0 + 1.0 / ratio_mass)
    )
    denominator = contract("...j,...ij->...i", mole_fractions, phi)
    return jnp.sum(mole_fractions * properties / denominator, axis=-1)


class MixtureAveragedTransportPlan(StrictModule, NonTrainableState):
    """Mixture-averaged gas transport with conservative correction velocity."""

    gas_model: ReactingGasModel
    reference_binary_diffusion: Array
    reference_species_viscosity: Array
    reference_species_conductivity: Array
    reference_temperature: float = eqx.field(static=True)
    reference_pressure: float = eqx.field(static=True)
    diffusion_temperature_exponent: float = eqx.field(static=True)
    viscosity_temperature_exponent: float = eqx.field(static=True)
    conductivity_temperature_exponent: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        gas_model: ReactingGasModel,
        binary_diffusion_coefficients: ArrayLike,
        species_viscosities: ArrayLike,
        species_thermal_conductivities: ArrayLike,
        /,
        *,
        reference_temperature: float = 300.0,
        reference_pressure: float = 101325.0,
        diffusion_temperature_exponent: float = 1.75,
        viscosity_temperature_exponent: float = 0.7,
        conductivity_temperature_exponent: float = 0.7,
        conservation_tolerance: float = 1.0e-10,
    ):
        if not isinstance(gas_model, ReactingGasModel):
            raise TypeError("gas_model must be ReactingGasModel.")
        species_count = gas_model.schema.species_count
        if species_count < 2:
            raise ValueError("Mixture transport requires at least two species.")
        diffusion = _validated_binary_diffusion(
            binary_diffusion_coefficients, species_count
        )
        viscosity = _validated_species_properties(
            species_viscosities, species_count, "species_viscosities"
        )
        conductivity = _validated_species_properties(
            species_thermal_conductivities,
            species_count,
            "species_thermal_conductivities",
        )
        scalars = tuple(
            float(value)
            for value in (
                reference_temperature,
                reference_pressure,
                diffusion_temperature_exponent,
                viscosity_temperature_exponent,
                conductivity_temperature_exponent,
                conservation_tolerance,
            )
        )
        if (
            any(not isfinite(value) for value in scalars)
            or scalars[0] <= 0.0
            or scalars[1] <= 0.0
            or scalars[5] <= 0.0
        ):
            raise ValueError("Transport references, exponents, or tolerance are invalid.")
        self.gas_model = gas_model
        self.reference_binary_diffusion = jnp.asarray(diffusion)
        self.reference_species_viscosity = jnp.asarray(viscosity)
        self.reference_species_conductivity = jnp.asarray(conductivity)
        self.reference_temperature = scalars[0]
        self.reference_pressure = scalars[1]
        self.diffusion_temperature_exponent = scalars[2]
        self.viscosity_temperature_exponent = scalars[3]
        self.conductivity_temperature_exponent = scalars[4]
        self.conservation_tolerance = scalars[5]
        self.transport_id = canonical_fingerprint(
            {
                "kind": "mixture-averaged-reacting-transport",
                "gas_model": gas_model.model_id,
                "binary_diffusion": array_tree_fingerprint(diffusion),
                "species_viscosity": array_tree_fingerprint(viscosity),
                "species_conductivity": array_tree_fingerprint(conductivity),
                "reference_temperature": scalars[0],
                "reference_pressure": scalars[1],
                "exponents": list(scalars[2:5]),
                "conservation_tolerance": scalars[5],
            }
        )

    def binary_diffusion(self, temperature: Array, pressure: Array, /) -> Array:
        scale = (
            temperature / self.reference_temperature
        ) ** self.diffusion_temperature_exponent * (self.reference_pressure / pressure)
        return scale[..., None, None] * self.reference_binary_diffusion

    def species_properties(self, temperature: Array, /) -> tuple[Array, Array]:
        ratio = temperature / self.reference_temperature
        viscosity = self.reference_species_viscosity * (
            ratio[..., None] ** self.viscosity_temperature_exponent
        )
        conductivity = self.reference_species_conductivity * (
            ratio[..., None] ** self.conductivity_temperature_exponent
        )
        return viscosity, conductivity

    def evaluate(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        density: ArrayLike,
        mass_fractions: ArrayLike,
        mass_fraction_gradient: ArrayLike,
        /,
        *,
        temperature_gradient: ArrayLike | None = None,
    ) -> ReactiveTransportEvaluation:
        temperature_ = jnp.asarray(temperature)
        pressure_ = jnp.asarray(pressure, dtype=temperature_.dtype)
        density_ = jnp.asarray(density, dtype=temperature_.dtype)
        mass = jnp.asarray(mass_fractions, dtype=temperature_.dtype)
        gradient = jnp.asarray(mass_fraction_gradient, dtype=temperature_.dtype)
        cell_shape = temperature_.shape
        species_count = self.gas_model.schema.species_count
        if pressure_.shape not in ((), cell_shape) or density_.shape not in (
            (),
            cell_shape,
        ):
            raise ValueError(
                "pressure and density must be scalar or match temperature cells."
            )
        pressure_ = jnp.broadcast_to(pressure_, cell_shape)
        density_ = jnp.broadcast_to(density_, cell_shape)
        if mass.shape != cell_shape + (species_count,):
            raise ValueError("mass_fractions has an invalid shape.")
        if (
            gradient.ndim != mass.ndim + 1
            or gradient.shape[:-2] != cell_shape
            or (gradient.shape[-2] != species_count)
        ):
            raise ValueError(
                "mass_fraction_gradient must have trailing species and spatial axes."
            )
        dimension = gradient.shape[-1]
        if temperature_gradient is None:
            temperature_gradient_ = jnp.zeros(cell_shape + (dimension,), dtype=mass.dtype)
        else:
            temperature_gradient_ = jnp.asarray(
                temperature_gradient, dtype=temperature_.dtype
            )
            if temperature_gradient_.shape != cell_shape + (dimension,):
                raise ValueError("temperature_gradient has an invalid shape.")
        thermo = self.gas_model.evaluate_pressure(temperature_, pressure_, mass)
        binary = self.binary_diffusion(temperature_, pressure_)
        mole = thermo.mole_fractions
        off_diagonal = ~jnp.eye(species_count, dtype=bool)
        resistance = jnp.where(
            off_diagonal,
            mole[..., None, :] / binary,
            0.0,
        )
        denominator = jnp.sum(resistance, axis=-1)
        mixture_diffusion = (1.0 - mass) / denominator
        raw_flux = -density_[..., None, None] * mixture_diffusion[..., :, None] * gradient
        correction = jnp.sum(raw_flux, axis=-2)
        species_flux = raw_flux - mass[..., :, None] * correction[..., None, :]
        velocities = species_flux / (density_[..., None, None] * mass[..., :, None])
        viscosity_species, conductivity_species = self.species_properties(temperature_)
        viscosity = _wilke_mixture(
            mole, viscosity_species, self.gas_model.schema.molar_masses
        )
        conductivity = _wilke_mixture(
            mole, conductivity_species, self.gas_model.schema.molar_masses
        )
        conductive_heat = -conductivity[..., None] * temperature_gradient_
        species_thermo = self.gas_model.thermodynamics.evaluate(temperature_)
        species_enthalpy = (
            species_thermo.molar_enthalpy + self.gas_model.formation_molar_enthalpies
        ) / self.gas_model.schema.molar_masses
        enthalpy_flux = contract("...sd,...s->...d", species_flux, species_enthalpy)
        total_heat = conductive_heat + enthalpy_flux
        net_mass = jnp.sum(species_flux, axis=-2)
        composition_residual = jnp.sum(gradient, axis=-2)
        flux_scale = jnp.maximum(jnp.max(jnp.abs(species_flux), axis=(-2, -1)), 1.0)
        valid = (
            thermo.successful
            & jnp.isfinite(density_)
            & (density_ > 0.0)
            & jnp.all(jnp.isfinite(gradient), axis=(-2, -1))
            & jnp.all(
                jnp.isfinite(jnp.where(off_diagonal, binary, 0.0)),
                axis=(-2, -1),
            )
            & jnp.all(denominator > 0.0, axis=-1)
            & jnp.all(jnp.isfinite(species_flux), axis=(-2, -1))
            & jnp.all(
                jnp.abs(net_mass) <= self.conservation_tolerance * flux_scale[..., None],
                axis=-1,
            )
            & jnp.all(
                jnp.abs(composition_residual) <= self.gas_model.composition_tolerance,
                axis=-1,
            )
        )
        return ReactiveTransportEvaluation(
            viscosity,
            conductivity,
            binary,
            mixture_diffusion,
            velocities,
            species_flux,
            conductive_heat,
            enthalpy_flux,
            total_heat,
            net_mass,
            composition_residual,
            valid,
            self.transport_id,
        )


class StefanMaxwellTransportPlan(StrictModule, NonTrainableState):
    """Bounded dense Stefan-Maxwell research-tier transport solve."""

    base: MixtureAveragedTransportPlan
    maximum_species: int = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    support_tier: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        gas_model: ReactingGasModel,
        binary_diffusion_coefficients: ArrayLike,
        species_viscosities: ArrayLike,
        species_thermal_conductivities: ArrayLike,
        /,
        *,
        maximum_species: int = 16,
        maximum_condition: float = 1.0e10,
        reference_temperature: float = 300.0,
        reference_pressure: float = 101325.0,
        diffusion_temperature_exponent: float = 1.75,
        viscosity_temperature_exponent: float = 0.7,
        conductivity_temperature_exponent: float = 0.7,
        conservation_tolerance: float = 1.0e-10,
    ):
        bound = int(maximum_species)
        condition = float(maximum_condition)
        species_count = gas_model.schema.species_count
        if bound < 2 or species_count < 2 or species_count > bound:
            raise ValueError(
                "Stefan-Maxwell research tier requires 2..maximum_species species."
            )
        if not isfinite(condition) or condition <= 1.0:
            raise ValueError("maximum_condition must be finite and greater than one.")
        base = MixtureAveragedTransportPlan(
            gas_model,
            binary_diffusion_coefficients,
            species_viscosities,
            species_thermal_conductivities,
            reference_temperature=reference_temperature,
            reference_pressure=reference_pressure,
            diffusion_temperature_exponent=diffusion_temperature_exponent,
            viscosity_temperature_exponent=viscosity_temperature_exponent,
            conductivity_temperature_exponent=conductivity_temperature_exponent,
            conservation_tolerance=conservation_tolerance,
        )
        self.base = base
        self.maximum_species = bound
        self.maximum_condition = condition
        self.support_tier = "research"
        self.transport_id = canonical_fingerprint(
            {
                "kind": "bounded-stefan-maxwell-transport",
                "base": base.transport_id,
                "maximum_species": bound,
                "maximum_condition": condition,
                "support_tier": "research",
            }
        )

    @property
    def gas_model(self) -> ReactingGasModel:
        return self.base.gas_model

    @property
    def conservation_tolerance(self) -> float:
        return self.base.conservation_tolerance

    def binary_diffusion(self, temperature: Array, pressure: Array, /) -> Array:
        return self.base.binary_diffusion(temperature, pressure)

    def evaluate(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        density: ArrayLike,
        mass_fractions: ArrayLike,
        mass_fraction_gradient: ArrayLike,
        /,
        *,
        temperature_gradient: ArrayLike | None = None,
    ) -> StefanMaxwellTransportEvaluation:
        mixture = self.base.evaluate(
            temperature,
            pressure,
            density,
            mass_fractions,
            mass_fraction_gradient,
            temperature_gradient=temperature_gradient,
        )
        temperature_ = jnp.asarray(temperature)
        pressure_ = jnp.asarray(pressure, dtype=temperature_.dtype)
        density_ = jnp.asarray(density, dtype=temperature_.dtype)
        mass = jnp.asarray(mass_fractions, dtype=temperature_.dtype)
        gradient = jnp.asarray(mass_fraction_gradient, dtype=temperature_.dtype)
        species_count = self.gas_model.schema.species_count
        molar_mass = self.gas_model.schema.molar_masses
        mixture_molar_mass = self.gas_model.mixture_molar_mass(mass)
        mole = mass * mixture_molar_mass[..., None] / molar_mass
        reciprocal_mixture_mass = 1.0 / mixture_molar_mass
        reciprocal_gradient = contract("...sd,s->...d", gradient, 1.0 / molar_mass)
        mole_gradient = (
            gradient / (molar_mass[..., None] * reciprocal_mixture_mass[..., None, None])
            - mole[..., :, None]
            * reciprocal_gradient[..., None, :]
            / reciprocal_mixture_mass[..., None, None]
        )
        binary = self.binary_diffusion(temperature_, pressure_)
        off_diagonal = ~jnp.eye(species_count, dtype=bool)
        matrix = jnp.where(
            off_diagonal,
            -mole[..., None, :] / binary,
            0.0,
        )
        diagonal = jnp.sum(
            jnp.where(off_diagonal, mole[..., None, :] / binary, 0.0), axis=-1
        )
        indices = jnp.arange(species_count)
        matrix = matrix.at[..., indices, indices].set(diagonal)
        matrix = matrix.at[..., -1, :].set(mass)
        right = -mole_gradient / mole[..., :, None]
        right = right.at[..., -1, :].set(0.0)
        inverse = dense_inverse(matrix)
        velocities = contract("...ij,...jd->...id", inverse, right)
        species_flux = density_[..., None, None] * mass[..., :, None] * velocities
        net_mass = jnp.sum(species_flux, axis=-2)
        residual = contract("...ij,...jd->...id", matrix, velocities) - right
        matrix_norm = jnp.max(jnp.sum(jnp.abs(matrix), axis=-1), axis=-1)
        inverse_norm = jnp.max(jnp.sum(jnp.abs(inverse), axis=-1), axis=-1)
        condition = matrix_norm * inverse_norm
        residual_scale = jnp.maximum(jnp.max(jnp.abs(right), axis=(-2, -1)), 1.0)
        mass_scale = jnp.maximum(jnp.max(jnp.abs(species_flux), axis=(-2, -1)), 1.0)
        successful = (
            mixture.successful
            & self.gas_model.composition_valid(mass)
            & jnp.all(mass > 0.0, axis=-1)
            & jnp.all(jnp.isfinite(matrix), axis=(-2, -1))
            & jnp.all(jnp.isfinite(velocities), axis=(-2, -1))
            & jnp.isfinite(condition)
            & (condition <= self.maximum_condition)
            & jnp.all(
                jnp.abs(residual)
                <= self.conservation_tolerance * residual_scale[..., None, None],
                axis=(-2, -1),
            )
            & jnp.all(
                jnp.abs(net_mass) <= self.conservation_tolerance * mass_scale[..., None],
                axis=-1,
            )
        )
        species_thermo = self.gas_model.thermodynamics.evaluate(temperature_)
        species_enthalpy = (
            species_thermo.molar_enthalpy + self.gas_model.formation_molar_enthalpies
        ) / molar_mass
        enthalpy_flux = contract("...sd,...s->...d", species_flux, species_enthalpy)
        total_heat = mixture.conductive_heat_flux + enthalpy_flux
        evidence = StefanMaxwellEvidence(
            matrix,
            right,
            residual,
            net_mass,
            condition,
            successful,
            self.transport_id,
        )
        return StefanMaxwellTransportEvaluation(
            mixture.dynamic_viscosity,
            mixture.thermal_conductivity,
            binary,
            mixture.mixture_diffusion_coefficients,
            velocities,
            species_flux,
            mixture.conductive_heat_flux,
            enthalpy_flux,
            total_heat,
            net_mass,
            mixture.composition_residual,
            successful,
            self.transport_id,
            evidence,
        )


__all__ = [
    "MixtureAveragedTransportPlan",
    "ReactiveTransportEvaluation",
    "StefanMaxwellEvidence",
    "StefanMaxwellTransportEvaluation",
    "StefanMaxwellTransportPlan",
]
