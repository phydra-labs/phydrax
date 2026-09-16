#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class GasTransportPropertyEvaluation(StrictModule):
    species_viscosity: Array
    species_thermal_conductivity: Array
    binary_diffusion_coefficients: Array
    viscosity_relative_error_bound: Array
    conductivity_relative_error_bound: Array
    diffusion_relative_error_bound: Array
    supported: Array
    finite: Array
    successful: Array
    property_id: str = eqx.field(static=True)


class AbstractGasTransportPropertyPlan(StrictModule, NonTrainableState, abc.ABC):
    species_count: int = eqx.field(static=True)
    property_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def evaluate(
        self, temperature: ArrayLike, pressure: ArrayLike, /
    ) -> GasTransportPropertyEvaluation:
        raise NotImplementedError


def _positive_vector(values: ArrayLike, size: int, name: str, /) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size,) or np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain one finite positive value per species.")
    return array


def _binary_matrix(values: ArrayLike, size: int, name: str, /) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must be species-square.")
    off_diagonal = ~np.eye(size, dtype=bool)
    if (
        np.any(~np.isfinite(matrix[off_diagonal]))
        or np.any(matrix[off_diagonal] <= 0.0)
        or not np.allclose(
            matrix[off_diagonal], matrix.T[off_diagonal], rtol=1.0e-12, atol=0.0
        )
    ):
        raise ValueError(f"{name} must be finite, positive off-diagonal, and symmetric.")
    result = matrix.copy()
    np.fill_diagonal(result, np.inf)
    return result


def _evaluation(
    viscosity: Array,
    conductivity: Array,
    diffusion: Array,
    viscosity_error: ArrayLike,
    conductivity_error: ArrayLike,
    diffusion_error: ArrayLike,
    supported: Array,
    property_id: str,
    /,
) -> GasTransportPropertyEvaluation:
    finite = (
        jnp.all(jnp.isfinite(viscosity), axis=-1)
        & jnp.all(jnp.isfinite(conductivity), axis=-1)
        & jnp.all(
            jnp.isfinite(
                jnp.where(
                    jnp.eye(diffusion.shape[-1], dtype=bool),
                    0.0,
                    diffusion,
                )
            ),
            axis=(-2, -1),
        )
    )
    positive = (
        jnp.all(viscosity > 0.0, axis=-1)
        & jnp.all(conductivity > 0.0, axis=-1)
        & jnp.all(
            jnp.where(jnp.eye(diffusion.shape[-1], dtype=bool), True, diffusion > 0.0),
            axis=(-2, -1),
        )
    )
    return GasTransportPropertyEvaluation(
        viscosity,
        conductivity,
        diffusion,
        jnp.asarray(viscosity_error, dtype=viscosity.dtype),
        jnp.asarray(conductivity_error, dtype=viscosity.dtype),
        jnp.asarray(diffusion_error, dtype=viscosity.dtype),
        supported,
        finite,
        supported & finite & positive,
        property_id,
    )


class ReferencePowerLawGasTransportPlan(AbstractGasTransportPropertyPlan):
    reference_binary_diffusion: Array
    reference_species_viscosity: Array
    reference_species_conductivity: Array
    reference_temperature: float = eqx.field(static=True)
    reference_pressure: float = eqx.field(static=True)
    diffusion_temperature_exponent: float = eqx.field(static=True)
    viscosity_temperature_exponent: float = eqx.field(static=True)
    conductivity_temperature_exponent: float = eqx.field(static=True)

    def __init__(
        self,
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
    ):
        viscosity = np.asarray(species_viscosities, dtype=float)
        count = int(viscosity.size) if viscosity.ndim == 1 else 0
        if count < 2:
            raise ValueError("Gas transport requires at least two species.")
        viscosity = _positive_vector(viscosity, count, "species_viscosities")
        conductivity = _positive_vector(
            species_thermal_conductivities,
            count,
            "species_thermal_conductivities",
        )
        diffusion = _binary_matrix(
            binary_diffusion_coefficients, count, "binary_diffusion_coefficients"
        )
        scalars = tuple(
            float(value)
            for value in (
                reference_temperature,
                reference_pressure,
                diffusion_temperature_exponent,
                viscosity_temperature_exponent,
                conductivity_temperature_exponent,
            )
        )
        if any(not isfinite(value) for value in scalars) or min(scalars[:2]) <= 0.0:
            raise ValueError("Transport references and exponents are invalid.")
        self.species_count = count
        self.reference_binary_diffusion = jnp.asarray(diffusion)
        self.reference_species_viscosity = jnp.asarray(viscosity)
        self.reference_species_conductivity = jnp.asarray(conductivity)
        self.reference_temperature, self.reference_pressure = scalars[:2]
        (
            self.diffusion_temperature_exponent,
            self.viscosity_temperature_exponent,
            self.conductivity_temperature_exponent,
        ) = scalars[2:]
        self.property_id = canonical_fingerprint(
            {
                "kind": "reference-power-law-gas-transport",
                "arrays": array_tree_fingerprint(
                    {
                        "diffusion": diffusion,
                        "viscosity": viscosity,
                        "conductivity": conductivity,
                    }
                ),
                "reference_temperature": scalars[0],
                "reference_pressure": scalars[1],
                "exponents": list(scalars[2:]),
            }
        )

    def evaluate(self, temperature, pressure, /):
        temperature_, pressure_ = jnp.broadcast_arrays(
            jnp.asarray(temperature), jnp.asarray(pressure)
        )
        ratio = temperature_ / self.reference_temperature
        viscosity = self.reference_species_viscosity * (
            ratio[..., None] ** self.viscosity_temperature_exponent
        )
        conductivity = self.reference_species_conductivity * (
            ratio[..., None] ** self.conductivity_temperature_exponent
        )
        diffusion = (
            self.reference_binary_diffusion
            * (ratio[..., None, None] ** self.diffusion_temperature_exponent)
            * (self.reference_pressure / pressure_[..., None, None])
        )
        supported = (
            jnp.isfinite(temperature_)
            & (temperature_ > 0.0)
            & jnp.isfinite(pressure_)
            & (pressure_ > 0.0)
        )
        return _evaluation(
            viscosity,
            conductivity,
            diffusion,
            0.0,
            0.0,
            0.0,
            supported,
            self.property_id,
        )


def _horner(coefficients: Array, coordinate: Array, /) -> Array:
    result = jnp.zeros(coordinate.shape + coefficients.shape[:-1], dtype=coordinate.dtype)
    for index in range(int(coefficients.shape[-1]) - 1, -1, -1):
        result = result * coordinate[..., None] + coefficients[..., index]
    return result


class LogPolynomialGasTransportPlan(AbstractGasTransportPropertyPlan):
    viscosity_coefficients: Array
    conductivity_coefficients: Array
    diffusion_coefficients: Array
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    reference_pressure: float = eqx.field(static=True)
    viscosity_error_bound: float = eqx.field(static=True)
    conductivity_error_bound: float = eqx.field(static=True)
    diffusion_error_bound: float = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        viscosity_coefficients: ArrayLike,
        conductivity_coefficients: ArrayLike,
        diffusion_coefficients: ArrayLike,
        temperature_bounds: tuple[float, float],
        /,
        *,
        reference_pressure: float,
        relative_error_bounds: tuple[float, float, float],
        reference_id: str,
    ):
        viscosity = np.asarray(viscosity_coefficients, dtype=float)
        conductivity = np.asarray(conductivity_coefficients, dtype=float)
        diffusion = np.asarray(diffusion_coefficients, dtype=float)
        if viscosity.ndim != 2 or viscosity.shape[0] < 2:
            raise ValueError("Viscosity log-polynomials must be species-by-coefficient.")
        count = int(viscosity.shape[0])
        if (
            conductivity.shape != viscosity.shape
            or diffusion.ndim != 3
            or diffusion.shape[:2] != (count, count)
            or diffusion.shape[-1] != viscosity.shape[-1]
            or np.any(~np.isfinite(viscosity))
            or np.any(~np.isfinite(conductivity))
        ):
            raise ValueError("Transport log-polynomial coefficient shapes are invalid.")
        off_diagonal = ~np.eye(count, dtype=bool)
        if np.any(~np.isfinite(diffusion[off_diagonal])):
            raise ValueError("Off-diagonal diffusion polynomials must be finite.")
        lower, upper = map(float, temperature_bounds)
        pressure = float(reference_pressure)
        errors = tuple(float(value) for value in relative_error_bounds)
        identifier = str(reference_id).strip()
        if (
            not 0.0 < lower < upper
            or not isfinite(pressure)
            or pressure <= 0.0
            or any(not isfinite(value) or value < 0.0 for value in errors)
            or not identifier
        ):
            raise ValueError(
                "Polynomial support, pressure, errors, or reference ID is invalid."
            )
        diffusion = diffusion.copy()
        diffusion[np.eye(count, dtype=bool)] = 0.0
        self.species_count = count
        self.viscosity_coefficients = jnp.asarray(viscosity)
        self.conductivity_coefficients = jnp.asarray(conductivity)
        self.diffusion_coefficients = jnp.asarray(diffusion)
        self.minimum_temperature, self.maximum_temperature = lower, upper
        self.reference_pressure = pressure
        (
            self.viscosity_error_bound,
            self.conductivity_error_bound,
            self.diffusion_error_bound,
        ) = errors
        self.reference_id = identifier
        self.property_id = canonical_fingerprint(
            {
                "kind": "log-polynomial-gas-transport",
                "arrays": array_tree_fingerprint(
                    {
                        "viscosity": viscosity,
                        "conductivity": conductivity,
                        "diffusion": diffusion,
                    }
                ),
                "temperature_bounds": [lower, upper],
                "reference_pressure": pressure,
                "relative_error_bounds": list(errors),
                "reference": identifier,
            }
        )

    def evaluate(self, temperature, pressure, /):
        temperature_, pressure_ = jnp.broadcast_arrays(
            jnp.asarray(temperature), jnp.asarray(pressure)
        )
        safe_temperature = jnp.maximum(temperature_, jnp.finfo(temperature_.dtype).tiny)
        log_temperature = jnp.log(safe_temperature)
        viscosity = jnp.exp(_horner(self.viscosity_coefficients, log_temperature))
        conductivity = jnp.exp(_horner(self.conductivity_coefficients, log_temperature))
        log_diffusion = _horner(
            self.diffusion_coefficients.reshape(
                (self.species_count * self.species_count, -1)
            ),
            log_temperature,
        ).reshape(temperature_.shape + (self.species_count, self.species_count))
        diffusion = jnp.exp(log_diffusion) * (
            self.reference_pressure / pressure_[..., None, None]
        )
        diffusion = jnp.where(jnp.eye(self.species_count, dtype=bool), jnp.inf, diffusion)
        supported = (
            jnp.isfinite(temperature_)
            & (temperature_ >= self.minimum_temperature)
            & (temperature_ <= self.maximum_temperature)
            & jnp.isfinite(pressure_)
            & (pressure_ > 0.0)
        )
        return _evaluation(
            viscosity,
            conductivity,
            diffusion,
            self.viscosity_error_bound,
            self.conductivity_error_bound,
            self.diffusion_error_bound,
            supported,
            self.property_id,
        )


class KineticTheoryGasTransportPlan(AbstractGasTransportPropertyPlan):
    molar_masses_g_mol: Array
    collision_diameters_angstrom: Array
    well_depths_kelvin: Array
    eucken_factors: Array
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)

    def __init__(
        self,
        molar_masses_g_mol: ArrayLike,
        collision_diameters_angstrom: ArrayLike,
        well_depths_kelvin: ArrayLike,
        eucken_factors: ArrayLike,
        temperature_bounds: tuple[float, float],
        /,
    ):
        masses = np.asarray(molar_masses_g_mol, dtype=float)
        count = int(masses.size) if masses.ndim == 1 else 0
        if count < 2:
            raise ValueError("Kinetic gas transport requires at least two species.")
        masses = _positive_vector(masses, count, "molar_masses_g_mol")
        diameters = _positive_vector(
            collision_diameters_angstrom, count, "collision_diameters_angstrom"
        )
        wells = _positive_vector(well_depths_kelvin, count, "well_depths_kelvin")
        eucken = _positive_vector(eucken_factors, count, "eucken_factors")
        lower, upper = map(float, temperature_bounds)
        if not 0.0 < lower < upper:
            raise ValueError("Kinetic transport temperature bounds are invalid.")
        self.species_count = count
        self.molar_masses_g_mol = jnp.asarray(masses)
        self.collision_diameters_angstrom = jnp.asarray(diameters)
        self.well_depths_kelvin = jnp.asarray(wells)
        self.eucken_factors = jnp.asarray(eucken)
        self.minimum_temperature, self.maximum_temperature = lower, upper
        self.property_id = canonical_fingerprint(
            {
                "kind": "lennard-jones-kinetic-gas-transport",
                "arrays": array_tree_fingerprint(
                    {
                        "molar_mass_g_mol": masses,
                        "diameter_angstrom": diameters,
                        "well_depth_kelvin": wells,
                        "eucken_factor": eucken,
                    }
                ),
                "temperature_bounds": [lower, upper],
                "collision_integral": "neufeld-style",
            }
        )

    @staticmethod
    def _omega_viscosity(reduced_temperature):
        return (
            1.16145 * reduced_temperature**-0.14874
            + 0.52487 * jnp.exp(-0.7732 * reduced_temperature)
            + 2.16178 * jnp.exp(-2.43787 * reduced_temperature)
        )

    @staticmethod
    def _omega_diffusion(reduced_temperature):
        return (
            1.06036 * reduced_temperature**-0.15610
            + 0.19300 * jnp.exp(-0.47635 * reduced_temperature)
            + 1.03587 * jnp.exp(-1.52996 * reduced_temperature)
            + 1.76474 * jnp.exp(-3.89411 * reduced_temperature)
        )

    def evaluate(self, temperature, pressure, /):
        temperature_, pressure_ = jnp.broadcast_arrays(
            jnp.asarray(temperature), jnp.asarray(pressure)
        )
        safe_temperature = jnp.maximum(temperature_, jnp.finfo(temperature_.dtype).tiny)
        reduced = safe_temperature[..., None] / self.well_depths_kelvin
        omega_mu = self._omega_viscosity(reduced)
        viscosity = (
            2.6693e-6
            * jnp.sqrt(self.molar_masses_g_mol * safe_temperature[..., None])
            / (self.collision_diameters_angstrom**2 * omega_mu)
        )
        universal_gas_constant = 8.31446261815324
        conductivity = (
            viscosity
            * self.eucken_factors
            * universal_gas_constant
            / (self.molar_masses_g_mol * 1.0e-3)
        )
        pair_mass = jnp.sqrt(
            1.0 / self.molar_masses_g_mol[:, None]
            + 1.0 / self.molar_masses_g_mol[None, :]
        )
        pair_diameter = 0.5 * (
            self.collision_diameters_angstrom[:, None]
            + self.collision_diameters_angstrom[None, :]
        )
        pair_well = jnp.sqrt(
            self.well_depths_kelvin[:, None] * self.well_depths_kelvin[None, :]
        )
        omega_d = self._omega_diffusion(safe_temperature[..., None, None] / pair_well)
        pressure_atmosphere = pressure_ / 101325.0
        diffusion = (
            1.858e-7
            * safe_temperature[..., None, None] ** 1.5
            * pair_mass
            / (pressure_atmosphere[..., None, None] * pair_diameter**2 * omega_d)
        )
        diffusion = jnp.where(jnp.eye(self.species_count, dtype=bool), jnp.inf, diffusion)
        supported = (
            jnp.isfinite(temperature_)
            & (temperature_ >= self.minimum_temperature)
            & (temperature_ <= self.maximum_temperature)
            & jnp.isfinite(pressure_)
            & (pressure_ > 0.0)
        )
        return _evaluation(
            viscosity,
            conductivity,
            diffusion,
            0.0,
            0.0,
            0.0,
            supported,
            self.property_id,
        )


__all__ = [
    "AbstractGasTransportPropertyPlan",
    "GasTransportPropertyEvaluation",
    "KineticTheoryGasTransportPlan",
    "LogPolynomialGasTransportPlan",
    "ReferencePowerLawGasTransportPlan",
]
