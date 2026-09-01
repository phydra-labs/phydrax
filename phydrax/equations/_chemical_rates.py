#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._chemical_thermodynamics import UNIVERSAL_GAS_CONSTANT


class ChemicalRateKind(StrEnum):
    ARRHENIUS = "arrhenius"
    THIRD_BODY = "third_body"
    LINDEMANN = "lindemann"
    TROE = "troe"
    PLOG = "plog"
    CHEBYSHEV = "chebyshev"
    PHOTOLYSIS = "photolysis"
    BUTLER_VOLMER = "butler_volmer"
    SURFACE_COVERAGE = "surface_coverage"
    STICKING = "sticking"


class ChemicalRateRuntime(StrictModule):
    photolysis_rates: Array
    overpotential: Array

    def __init__(
        self,
        photolysis_rates: ArrayLike | None = None,
        overpotential: ArrayLike = 0.0,
    ):
        rates = (
            jnp.zeros((0,)) if photolysis_rates is None else jnp.asarray(photolysis_rates)
        )
        potential = jnp.asarray(overpotential)
        if rates.ndim != 1:
            raise ValueError("photolysis_rates must be one-dimensional.")
        self.photolysis_rates = rates
        self.overpotential = potential


class AbstractChemicalRatePlan(StrictModule, abc.ABC):
    kind: ChemicalRateKind = eqx.field(static=True)

    @abc.abstractmethod
    def evaluate(
        self,
        temperature: Array,
        pressure: Array,
        concentrations: Array,
        runtime: ChemicalRateRuntime,
        /,
    ) -> Array:
        raise NotImplementedError


class ArrheniusRatePlan(AbstractChemicalRatePlan):
    pre_exponential: Array
    temperature_exponent: Array
    activation_energy: Array

    def __init__(
        self,
        pre_exponential: ArrayLike,
        temperature_exponent: ArrayLike = 0.0,
        activation_energy: ArrayLike = 0.0,
        /,
    ):
        pre = jnp.asarray(pre_exponential)
        exponent = jnp.asarray(temperature_exponent, dtype=pre.dtype)
        activation = jnp.asarray(activation_energy, dtype=pre.dtype)
        if pre.shape != () or exponent.shape != () or activation.shape != ():
            raise ValueError("Arrhenius parameters must be scalar.")
        self.kind = ChemicalRateKind.ARRHENIUS
        self.pre_exponential = pre
        self.temperature_exponent = exponent
        self.activation_energy = activation

    def evaluate(self, temperature, pressure, concentrations, runtime, /):
        del pressure, concentrations, runtime
        valid = (
            jnp.isfinite(temperature)
            & (temperature > 0.0)
            & jnp.isfinite(self.pre_exponential)
            & (self.pre_exponential >= 0.0)
            & jnp.isfinite(self.temperature_exponent)
            & jnp.isfinite(self.activation_energy)
        )
        safe_temperature = jnp.where(valid, temperature, 1.0)
        rate = (
            self.pre_exponential
            * safe_temperature**self.temperature_exponent
            * jnp.exp(
                -self.activation_energy / (UNIVERSAL_GAS_CONSTANT * safe_temperature)
            )
        )
        return jnp.where(valid, rate, jnp.nan)


class ThirdBodyRatePlan(AbstractChemicalRatePlan):
    base: ArrheniusRatePlan
    efficiencies: Array

    def __init__(self, base: ArrheniusRatePlan, efficiencies: ArrayLike, /):
        if not isinstance(base, ArrheniusRatePlan):
            raise TypeError("base must be ArrheniusRatePlan.")
        values = jnp.asarray(efficiencies)
        if values.ndim != 1:
            raise ValueError("efficiencies must be one-dimensional.")
        self.kind = ChemicalRateKind.THIRD_BODY
        self.base = base
        self.efficiencies = values

    def effective_concentration(self, concentrations):
        if concentrations.shape[-1] != self.efficiencies.shape[0]:
            raise ValueError("Third-body efficiencies must match species axis.")
        return jnp.sum(concentrations * self.efficiencies, axis=-1)

    def evaluate(self, temperature, pressure, concentrations, runtime, /):
        return self.base.evaluate(
            temperature, pressure, concentrations, runtime
        ) * self.effective_concentration(concentrations)


class LindemannRatePlan(AbstractChemicalRatePlan):
    low_pressure: ArrheniusRatePlan
    high_pressure: ArrheniusRatePlan
    efficiencies: Array

    def __init__(self, low_pressure, high_pressure, efficiencies: ArrayLike, /):
        if not isinstance(low_pressure, ArrheniusRatePlan) or not isinstance(
            high_pressure, ArrheniusRatePlan
        ):
            raise TypeError("Falloff limits must be ArrheniusRatePlan objects.")
        values = jnp.asarray(efficiencies)
        if values.ndim != 1:
            raise ValueError("efficiencies must be one-dimensional.")
        self.kind = ChemicalRateKind.LINDEMANN
        self.low_pressure = low_pressure
        self.high_pressure = high_pressure
        self.efficiencies = values

    def reduced_pressure(self, temperature, pressure, concentrations, runtime):
        effective = jnp.sum(concentrations * self.efficiencies, axis=-1)
        low = self.low_pressure.evaluate(temperature, pressure, concentrations, runtime)
        high = self.high_pressure.evaluate(temperature, pressure, concentrations, runtime)
        reduced = low * effective / jnp.maximum(high, jnp.finfo(high.dtype).tiny)
        return reduced, high

    def evaluate(self, temperature, pressure, concentrations, runtime, /):
        reduced, high = self.reduced_pressure(
            temperature, pressure, concentrations, runtime
        )
        return high * reduced / (1.0 + reduced)


class TroeRatePlan(AbstractChemicalRatePlan):
    low_pressure: ArrheniusRatePlan
    high_pressure: ArrheniusRatePlan
    efficiencies: Array
    alpha: Array
    temperature_1: Array
    temperature_2: Array
    temperature_3: Array

    def __init__(
        self,
        low_pressure,
        high_pressure,
        efficiencies: ArrayLike,
        alpha: ArrayLike,
        temperature_1: ArrayLike,
        temperature_2: ArrayLike,
        temperature_3: ArrayLike,
        /,
    ):
        if not isinstance(low_pressure, ArrheniusRatePlan) or not isinstance(
            high_pressure, ArrheniusRatePlan
        ):
            raise TypeError("Falloff limits must be ArrheniusRatePlan objects.")
        efficiency_values = jnp.asarray(efficiencies)
        values = tuple(
            jnp.asarray(value)
            for value in (alpha, temperature_1, temperature_2, temperature_3)
        )
        if efficiency_values.ndim != 1 or any(value.shape != () for value in values):
            raise ValueError("Troe efficiencies/parameters have invalid shapes.")
        self.kind = ChemicalRateKind.TROE
        self.low_pressure = low_pressure
        self.high_pressure = high_pressure
        self.efficiencies = efficiency_values
        self.alpha, self.temperature_1, self.temperature_2, self.temperature_3 = values

    def reduced_pressure(self, temperature, pressure, concentrations, runtime):
        effective = jnp.sum(concentrations * self.efficiencies, axis=-1)
        low = self.low_pressure.evaluate(temperature, pressure, concentrations, runtime)
        high = self.high_pressure.evaluate(temperature, pressure, concentrations, runtime)
        reduced = low * effective / jnp.maximum(high, jnp.finfo(high.dtype).tiny)
        return reduced, high

    def evaluate(self, temperature, pressure, concentrations, runtime, /):
        reduced, high = self.reduced_pressure(
            temperature, pressure, concentrations, runtime
        )
        f_center = (
            (1.0 - self.alpha) * jnp.exp(-temperature / self.temperature_3)
            + self.alpha * jnp.exp(-temperature / self.temperature_1)
            + jnp.exp(-self.temperature_2 / temperature)
        )
        tiny = jnp.finfo(temperature.dtype).tiny
        log_center = jnp.log10(jnp.maximum(f_center, tiny))
        log_reduced = jnp.log10(jnp.maximum(reduced, tiny))
        c_value = -0.4 - 0.67 * log_center
        n_value = 0.75 - 1.27 * log_center
        numerator = log_reduced + c_value
        denominator = n_value - 0.14 * numerator
        log_falloff = log_center / (1.0 + (numerator / denominator) ** 2)
        return high * reduced / (1.0 + reduced) * 10.0**log_falloff


class PLogRatePlan(AbstractChemicalRatePlan):
    pressures: Array
    rates: tuple[ArrheniusRatePlan, ...]

    def __init__(self, pressures: ArrayLike, rates, /):
        pressure_values = np.asarray(pressures, dtype=float)
        rate_values = tuple(rates)
        if (
            pressure_values.ndim != 1
            or pressure_values.size < 2
            or np.any(~np.isfinite(pressure_values))
            or np.any(pressure_values <= 0.0)
            or np.any(np.diff(pressure_values) <= 0.0)
            or len(rate_values) != pressure_values.size
            or any(not isinstance(value, ArrheniusRatePlan) for value in rate_values)
        ):
            raise ValueError("PLOG pressures/rates are invalid.")
        self.kind = ChemicalRateKind.PLOG
        self.pressures = jnp.asarray(pressure_values)
        self.rates = rate_values

    def evaluate(self, temperature, pressure, concentrations, runtime, /):
        valid = jnp.isfinite(pressure) & (pressure > 0.0)
        safe_pressure = jnp.where(valid, pressure, self.pressures[0])
        index = jnp.clip(
            jnp.searchsorted(self.pressures, safe_pressure) - 1, 0, len(self.rates) - 2
        )
        all_rates = jnp.stack(
            [
                value.evaluate(temperature, pressure, concentrations, runtime)
                for value in self.rates
            ],
            axis=-1,
        )
        lower_rate = jnp.take_along_axis(all_rates, index[..., None], axis=-1)[..., 0]
        upper_rate = jnp.take_along_axis(all_rates, (index + 1)[..., None], axis=-1)[
            ..., 0
        ]
        lower_pressure = self.pressures[index]
        upper_pressure = self.pressures[index + 1]
        fraction = (jnp.log(safe_pressure) - jnp.log(lower_pressure)) / (
            jnp.log(upper_pressure) - jnp.log(lower_pressure)
        )
        log_rate = (1.0 - fraction) * jnp.log(
            jnp.maximum(lower_rate, jnp.finfo(lower_rate.dtype).tiny)
        ) + fraction * jnp.log(jnp.maximum(upper_rate, jnp.finfo(upper_rate.dtype).tiny))
        return jnp.where(valid, jnp.exp(log_rate), jnp.nan)


class ChebyshevRatePlan(AbstractChemicalRatePlan):
    coefficients: Array
    minimum_temperature: float = eqx.field(static=True)
    maximum_temperature: float = eqx.field(static=True)
    minimum_pressure: float = eqx.field(static=True)
    maximum_pressure: float = eqx.field(static=True)

    def __init__(
        self,
        coefficients: ArrayLike,
        minimum_temperature: float,
        maximum_temperature: float,
        minimum_pressure: float,
        maximum_pressure: float,
        /,
    ):
        values = np.asarray(coefficients, dtype=float)
        bounds = tuple(
            float(value)
            for value in (
                minimum_temperature,
                maximum_temperature,
                minimum_pressure,
                maximum_pressure,
            )
        )
        if (
            values.ndim != 2
            or values.size == 0
            or np.any(~np.isfinite(values))
            or not 0.0 < bounds[0] < bounds[1]
            or not 0.0 < bounds[2] < bounds[3]
        ):
            raise ValueError("Chebyshev rate inputs are invalid.")
        self.kind = ChemicalRateKind.CHEBYSHEV
        self.coefficients = jnp.asarray(values)
        (
            self.minimum_temperature,
            self.maximum_temperature,
            self.minimum_pressure,
            self.maximum_pressure,
        ) = bounds

    def evaluate(self, temperature, pressure, concentrations, runtime, /):
        del concentrations, runtime
        valid = (
            jnp.isfinite(temperature)
            & jnp.isfinite(pressure)
            & (temperature >= self.minimum_temperature)
            & (temperature <= self.maximum_temperature)
            & (pressure >= self.minimum_pressure)
            & (pressure <= self.maximum_pressure)
        )
        safe_temperature = jnp.where(valid, temperature, self.minimum_temperature)
        safe_pressure = jnp.where(valid, pressure, self.minimum_pressure)
        reciprocal_temperature = 1.0 / safe_temperature
        reduced_temperature = (
            2.0 * reciprocal_temperature
            - 1.0 / self.minimum_temperature
            - 1.0 / self.maximum_temperature
        ) / (1.0 / self.maximum_temperature - 1.0 / self.minimum_temperature)
        log_pressure = jnp.log10(safe_pressure)
        reduced_pressure = (
            2.0 * log_pressure
            - np.log10(self.minimum_pressure)
            - np.log10(self.maximum_pressure)
        ) / (np.log10(self.maximum_pressure) - np.log10(self.minimum_pressure))
        temperature_basis = _chebyshev_basis(
            reduced_temperature, self.coefficients.shape[0]
        )
        pressure_basis = _chebyshev_basis(reduced_pressure, self.coefficients.shape[1])
        log_rate = jnp.sum(
            temperature_basis[..., :, None]
            * self.coefficients
            * pressure_basis[..., None, :],
            axis=(-2, -1),
        )
        return jnp.where(valid, 10.0**log_rate, jnp.nan)


class PhotolysisRatePlan(AbstractChemicalRatePlan):
    channel: int = eqx.field(static=True)

    def __init__(self, channel: int, /):
        value = int(channel)
        if value < 0:
            raise ValueError("Photolysis channel must be nonnegative.")
        self.kind = ChemicalRateKind.PHOTOLYSIS
        self.channel = value

    def evaluate(self, temperature, pressure, concentrations, runtime, /):
        del temperature, pressure, concentrations
        if self.channel >= runtime.photolysis_rates.shape[0]:
            raise ValueError("Photolysis runtime does not provide the requested channel.")
        return runtime.photolysis_rates[self.channel]


class SurfaceCoverageRatePlan(AbstractChemicalRatePlan):
    base: ArrheniusRatePlan
    species_index: int = eqx.field(static=True)
    exponential_coefficient: Array
    power_exponent: Array
    activation_energy_coefficient: Array

    def __init__(
        self,
        base: ArrheniusRatePlan,
        species_index: int,
        /,
        *,
        exponential_coefficient: ArrayLike = 0.0,
        power_exponent: ArrayLike = 0.0,
        activation_energy_coefficient: ArrayLike = 0.0,
    ):
        if not isinstance(base, ArrheniusRatePlan):
            raise TypeError("base must be ArrheniusRatePlan.")
        index = int(species_index)
        values = tuple(
            jnp.asarray(value)
            for value in (
                exponential_coefficient,
                power_exponent,
                activation_energy_coefficient,
            )
        )
        if index < 0 or any(value.shape != () for value in values):
            raise ValueError("Surface coverage rate inputs are invalid.")
        self.kind = ChemicalRateKind.SURFACE_COVERAGE
        self.base = base
        self.species_index = index
        (
            self.exponential_coefficient,
            self.power_exponent,
            self.activation_energy_coefficient,
        ) = values

    def evaluate(self, temperature, pressure, concentrations, runtime, /):
        if self.species_index >= concentrations.shape[-1]:
            raise ValueError("Coverage species index exceeds concentration axis.")
        coverage = concentrations[..., self.species_index]
        positive = coverage > 0.0
        safe = jnp.where(positive, coverage, 1.0)
        modifier = (
            jnp.exp(self.exponential_coefficient * coverage)
            * safe**self.power_exponent
            * jnp.exp(
                -self.activation_energy_coefficient
                * coverage
                / (UNIVERSAL_GAS_CONSTANT * temperature)
            )
        )
        requires_positive = self.power_exponent != 0.0
        valid = (~requires_positive) | positive
        return jnp.where(
            valid,
            self.base.evaluate(temperature, pressure, concentrations, runtime) * modifier,
            0.0,
        )


class StickingRatePlan(AbstractChemicalRatePlan):
    sticking_coefficient: Array
    molar_mass: Array

    def __init__(
        self,
        sticking_coefficient: ArrayLike,
        molar_mass: ArrayLike,
        /,
    ):
        coefficient = jnp.asarray(sticking_coefficient)
        mass = jnp.asarray(molar_mass, dtype=coefficient.dtype)
        if coefficient.shape != () or mass.shape != ():
            raise ValueError("Sticking parameters must be scalar.")
        self.kind = ChemicalRateKind.STICKING
        self.sticking_coefficient = coefficient
        self.molar_mass = mass

    def evaluate(self, temperature, pressure, concentrations, runtime, /):
        del pressure, concentrations, runtime
        valid = (
            jnp.isfinite(temperature)
            & (temperature > 0.0)
            & jnp.isfinite(self.sticking_coefficient)
            & (self.sticking_coefficient >= 0.0)
            & (self.sticking_coefficient <= 1.0)
            & jnp.isfinite(self.molar_mass)
            & (self.molar_mass > 0.0)
        )
        thermal_speed = jnp.sqrt(
            UNIVERSAL_GAS_CONSTANT * temperature / (2.0 * jnp.pi * self.molar_mass)
        )
        return jnp.where(valid, self.sticking_coefficient * thermal_speed, jnp.nan)


class ButlerVolmerRatePlan(AbstractChemicalRatePlan):
    exchange_rate: Array
    transfer_coefficient: Array
    electron_count: int = eqx.field(static=True)
    direction: int = eqx.field(static=True)

    def __init__(
        self,
        exchange_rate: ArrayLike,
        transfer_coefficient: ArrayLike,
        electron_count: int,
        /,
        *,
        direction: int = 1,
    ):
        exchange = jnp.asarray(exchange_rate)
        coefficient = jnp.asarray(transfer_coefficient, dtype=exchange.dtype)
        electrons = int(electron_count)
        direction_ = int(direction)
        if exchange.shape != () or coefficient.shape != ():
            raise ValueError("Butler-Volmer parameters must be scalar.")
        if electrons <= 0 or direction_ not in (-1, 1):
            raise ValueError("Butler-Volmer electron count/direction is invalid.")
        self.kind = ChemicalRateKind.BUTLER_VOLMER
        self.exchange_rate = exchange
        self.transfer_coefficient = coefficient
        self.electron_count = electrons
        self.direction = direction_

    def evaluate(self, temperature, pressure, concentrations, runtime, /):
        del pressure, concentrations
        coefficient = (
            self.transfer_coefficient
            if self.direction > 0
            else 1.0 - self.transfer_coefficient
        )
        exponent = (
            self.direction
            * coefficient
            * self.electron_count
            * 96485.33212
            * runtime.overpotential
            / (UNIVERSAL_GAS_CONSTANT * temperature)
        )
        valid = (
            jnp.isfinite(temperature)
            & (temperature > 0.0)
            & jnp.isfinite(self.exchange_rate)
            & (self.exchange_rate >= 0.0)
            & jnp.isfinite(self.transfer_coefficient)
            & (self.transfer_coefficient > 0.0)
            & (self.transfer_coefficient < 1.0)
            & jnp.isfinite(runtime.overpotential)
        )
        return jnp.where(valid, self.exchange_rate * jnp.exp(exponent), jnp.nan)


def _chebyshev_basis(value, count):
    terms = [jnp.ones_like(value)]
    if count > 1:
        terms.append(value)
    for _ in range(2, count):
        terms.append(2.0 * value * terms[-1] - terms[-2])
    return jnp.stack(terms, axis=-1)


__all__ = [
    "AbstractChemicalRatePlan",
    "ArrheniusRatePlan",
    "ButlerVolmerRatePlan",
    "ChemicalRateKind",
    "ChemicalRateRuntime",
    "ChebyshevRatePlan",
    "LindemannRatePlan",
    "PhotolysisRatePlan",
    "PLogRatePlan",
    "StickingRatePlan",
    "SurfaceCoverageRatePlan",
    "ThirdBodyRatePlan",
    "TroeRatePlan",
]
