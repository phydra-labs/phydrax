#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._interpolation import apply_gather_stencil, rectilinear_stencil
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._homogeneous_thermodynamics import HomogeneousHelmholtzPlan


_PLANCK_CONSTANT = 6.62607015e-34
_BOLTZMANN_CONSTANT = 1.380649e-23
_LIGHT_SPEED = 299792458.0
_RADIATION_CONSTANT = 7.565733250033928e-16


class RadiationCoefficientRole(StrEnum):
    ABSORPTION = "absorption"
    SCATTERING = "scattering"
    TRANSPORT = "transport"


class RadiationScaleContract(StrictModule, NonTrainableState):
    physical_light_speed: float = eqx.field(static=True)
    reduced_light_speed: float = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        reduced_light_speed: float,
        physical_light_speed: float = _LIGHT_SPEED,
    ) -> None:
        physical = float(physical_light_speed)
        reduced = float(reduced_light_speed)
        if (
            not np.isfinite(physical)
            or physical <= 0.0
            or not np.isfinite(reduced)
            or reduced <= 0.0
            or reduced > physical
        ):
            raise ValueError("Radiation light-speed contract is invalid.")
        self.physical_light_speed = physical
        self.reduced_light_speed = reduced
        self.contract_id = canonical_fingerprint(
            {
                "kind": "radiation-scale-contract",
                "physical_light_speed": physical,
                "reduced_light_speed": reduced,
            }
        )


class SpectralFrequencyGrid(StrictModule, NonTrainableState):
    frequency: Array
    quadrature_weight: Array
    grid_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequency: ArrayLike,
        quadrature_weight: ArrayLike,
        /,
    ) -> None:
        values = np.asarray(frequency, dtype=float)
        weights = np.asarray(quadrature_weight, dtype=float)
        if (
            values.ndim != 1
            or values.size < 2
            or weights.shape != values.shape
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or np.any(np.diff(values) <= 0.0)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("Spectral frequency grid and quadrature are invalid.")
        self.frequency = jnp.asarray(values)
        self.quadrature_weight = jnp.asarray(weights)
        self.grid_id = canonical_fingerprint(
            {
                "kind": "spectral-frequency-grid",
                "frequency": array_tree_fingerprint(values),
                "quadrature_weight": array_tree_fingerprint(weights),
            }
        )


class RadiationCoefficientEvaluation(StrictModule):
    coefficient: Array
    supported: Array
    table_id: str = eqx.field(static=True)


class RadiationCoefficientTable(StrictModule, NonTrainableState):
    temperature_axis: Array
    pressure_axis: Array
    frequency_grid: SpectralFrequencyGrid
    coefficient: Array
    role: RadiationCoefficientRole = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperature_axis: ArrayLike,
        pressure_axis: ArrayLike,
        frequency_grid: SpectralFrequencyGrid,
        coefficient: ArrayLike,
        role: RadiationCoefficientRole,
        /,
        *,
        provenance: str,
    ) -> None:
        temperature = np.asarray(temperature_axis, dtype=float)
        pressure = np.asarray(pressure_axis, dtype=float)
        values = np.asarray(coefficient, dtype=float)
        source = str(provenance)
        if not isinstance(frequency_grid, SpectralFrequencyGrid):
            raise TypeError("frequency_grid must be SpectralFrequencyGrid.")
        if not isinstance(role, RadiationCoefficientRole):
            raise TypeError("role must be RadiationCoefficientRole.")
        if (
            temperature.ndim != 1
            or pressure.ndim != 1
            or temperature.size < 2
            or pressure.size < 2
            or np.any(np.diff(temperature) <= 0.0)
            or np.any(np.diff(pressure) <= 0.0)
            or values.shape
            != (temperature.size, pressure.size, frequency_grid.frequency.size)
            or np.any(~np.isfinite(values))
            or np.any(values <= 0.0)
            or not source
        ):
            raise ValueError("Radiation coefficient table is invalid.")
        self.temperature_axis = jnp.asarray(temperature)
        self.pressure_axis = jnp.asarray(pressure)
        self.frequency_grid = frequency_grid
        self.coefficient = jnp.asarray(values)
        self.role = role
        self.provenance = source
        self.table_id = canonical_fingerprint(
            {
                "kind": "radiation-coefficient-table",
                "role": role.value,
                "temperature": array_tree_fingerprint(temperature),
                "pressure": array_tree_fingerprint(pressure),
                "frequency_grid": frequency_grid.grid_id,
                "coefficient": array_tree_fingerprint(values),
                "provenance": source,
            }
        )

    def evaluate(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        /,
    ) -> RadiationCoefficientEvaluation:
        temperature_value = jnp.asarray(temperature)
        pressure_value = jnp.asarray(pressure)
        shape = jnp.broadcast_shapes(temperature_value.shape, pressure_value.shape)
        frequency = jnp.broadcast_to(
            self.frequency_grid.frequency,
            shape + self.frequency_grid.frequency.shape,
        )
        query = jnp.stack(
            (
                jnp.broadcast_to(temperature_value, shape)[..., None]
                * jnp.ones_like(frequency),
                jnp.broadcast_to(pressure_value, shape)[..., None]
                * jnp.ones_like(frequency),
                frequency,
            ),
            axis=-1,
        )
        stencil = rectilinear_stencil(
            (
                self.temperature_axis,
                self.pressure_axis,
                self.frequency_grid.frequency,
            ),
            query,
            boundary=("constant", "constant", "constant"),
        )
        evaluated = apply_gather_stencil(self.coefficient.reshape((-1,)), stencil)
        return RadiationCoefficientEvaluation(
            evaluated.values,
            evaluated.support,
            self.table_id,
        )


class RadiationMeanEvaluation(StrictModule):
    planck_absorption: Array
    rosseland_transport: Array
    successful: Array
    coefficient_id: str = eqx.field(static=True)


def radiation_means(
    temperature: ArrayLike,
    absorption: RadiationCoefficientEvaluation,
    transport: RadiationCoefficientEvaluation,
    grid: SpectralFrequencyGrid,
    /,
) -> RadiationMeanEvaluation:
    temperature_value = jnp.asarray(temperature)
    frequency = grid.frequency.astype(temperature_value.dtype)
    exponent = (
        _PLANCK_CONSTANT
        * frequency
        / (_BOLTZMANN_CONSTANT * temperature_value[..., None])
    )
    denominator = jnp.expm1(exponent)
    planck = 2.0 * _PLANCK_CONSTANT * frequency**3 / _LIGHT_SPEED**2 / denominator
    planck_temperature_derivative = (
        planck
        * exponent
        * jnp.exp(exponent)
        / (temperature_value[..., None] * denominator)
    )
    weight = grid.quadrature_weight.astype(planck.dtype)
    planck_mean = jnp.sum(weight * absorption.coefficient * planck, axis=-1) / jnp.sum(
        weight * planck, axis=-1
    )
    rosseland_mean = jnp.sum(weight * planck_temperature_derivative, axis=-1) / jnp.sum(
        weight * planck_temperature_derivative / transport.coefficient,
        axis=-1,
    )
    successful = (
        jnp.all(absorption.supported, axis=-1)
        & jnp.all(transport.supported, axis=-1)
        & jnp.isfinite(planck_mean)
        & (planck_mean > 0.0)
        & jnp.isfinite(rosseland_mean)
        & (rosseland_mean > 0.0)
    )
    return RadiationMeanEvaluation(
        planck_mean,
        rosseland_mean,
        successful,
        canonical_fingerprint(
            {
                "kind": "radiation-means",
                "absorption": absorption.table_id,
                "transport": transport.table_id,
                "grid": grid.grid_id,
            }
        ),
    )


class RadiationMatterExchangeResult(StrictModule):
    radiation_energy_density: Array
    material_internal_energy_density: Array
    material_temperature: Array
    exchange: Array
    residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class RadiationMatterExchangePlan(StrictModule):
    thermodynamics: HomogeneousHelmholtzPlan
    scale: RadiationScaleContract
    absorption_coefficient: float = eqx.field(static=True)
    radiation_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        thermodynamics: HomogeneousHelmholtzPlan,
        scale: RadiationScaleContract,
        /,
        *,
        absorption_coefficient: float,
        radiation_constant: float = _RADIATION_CONSTANT,
    ) -> None:
        if not isinstance(thermodynamics, HomogeneousHelmholtzPlan):
            raise TypeError("thermodynamics must be HomogeneousHelmholtzPlan.")
        if not isinstance(scale, RadiationScaleContract):
            raise TypeError("scale must be RadiationScaleContract.")
        absorption = float(absorption_coefficient)
        constant = float(radiation_constant)
        if (
            not np.isfinite(absorption)
            or absorption <= 0.0
            or not np.isfinite(constant)
            or constant <= 0.0
        ):
            raise ValueError("Radiation exchange coefficients must be positive.")
        self.thermodynamics = thermodynamics
        self.scale = scale
        self.absorption_coefficient = absorption
        self.radiation_constant = constant
        self.plan_id = canonical_fingerprint(
            {
                "kind": "radiation-matter-exchange",
                "thermodynamics": thermodynamics.model_id,
                "scale": scale.contract_id,
                "absorption_coefficient": absorption,
                "radiation_constant": constant,
            }
        )

    def advance(
        self,
        species_mass_density: ArrayLike,
        radiation_energy_density: ArrayLike,
        material_internal_energy_density: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> RadiationMatterExchangeResult:
        species_density = jnp.asarray(species_mass_density)
        radiation_initial = jnp.asarray(radiation_energy_density)
        material_initial = jnp.asarray(material_internal_energy_density)
        step = jnp.asarray(step_size)
        total = radiation_initial + material_initial
        lower = jnp.maximum(
            jnp.asarray(0.0, dtype=total.dtype),
            total - material_initial * 10.0,
        )
        upper = jnp.maximum(total, jnp.asarray(0.0, dtype=total.dtype))
        coefficient = step * self.scale.reduced_light_speed * self.absorption_coefficient

        def residual(radiation):
            material = total - radiation
            thermal = self.thermodynamics.solve_density_energy(species_density, material)
            equilibrium = self.radiation_constant * thermal.state.temperature**4
            return radiation - radiation_initial - coefficient * (equilibrium - radiation)

        def body(_, bounds):
            low, high = bounds
            midpoint = 0.5 * (low + high)
            value = residual(midpoint)
            return (
                jnp.where(value < 0.0, midpoint, low),
                jnp.where(value < 0.0, high, midpoint),
            )

        low, high = jax.lax.fori_loop(0, 80, body, (lower, upper))
        radiation = 0.5 * (low + high)
        material = total - radiation
        thermal = self.thermodynamics.solve_density_energy(species_density, material)
        final_residual = residual(radiation)
        tolerance = 512.0 * jnp.finfo(total.dtype).eps * jnp.maximum(jnp.abs(total), 1.0)
        successful = (
            jnp.isfinite(step)
            & (step >= 0.0)
            & (radiation >= 0.0)
            & (material >= 0.0)
            & thermal.successful
            & jnp.isfinite(final_residual)
            & (jnp.abs(final_residual) <= tolerance)
        )
        return RadiationMatterExchangeResult(
            radiation,
            material,
            thermal.state.temperature,
            radiation - radiation_initial,
            final_residual,
            successful,
            self.plan_id,
        )


__all__ = [
    "RadiationCoefficientEvaluation",
    "RadiationCoefficientRole",
    "RadiationCoefficientTable",
    "RadiationMatterExchangePlan",
    "RadiationMatterExchangeResult",
    "RadiationMeanEvaluation",
    "RadiationScaleContract",
    "SpectralFrequencyGrid",
    "radiation_means",
]
