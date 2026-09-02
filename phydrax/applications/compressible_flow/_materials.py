#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._materials import AbstractThermodynamicMaterial


class EOSDerivativeCertificate(StrictModule, NonTrainableState):
    """Domain-bound derivative and inverse-consistency evidence for one EOS."""

    model_id: str = eqx.field(static=True)
    density_bounds: tuple[float, float] = eqx.field(static=True)
    temperature_bounds: tuple[float, float] = eqx.field(static=True)
    maximum_inverse_residual: float = eqx.field(static=True)
    maximum_derivative_residual: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        model_id: str,
        density_bounds: Sequence[float],
        temperature_bounds: Sequence[float],
        /,
        *,
        maximum_inverse_residual: float,
        maximum_derivative_residual: float,
        tolerance: float,
    ):
        identifier = str(model_id)
        density = tuple(float(value) for value in density_bounds)
        temperature = tuple(float(value) for value in temperature_bounds)
        inverse = float(maximum_inverse_residual)
        derivative = float(maximum_derivative_residual)
        tolerance_ = float(tolerance)
        if (
            not identifier
            or len(density) != 2
            or len(temperature) != 2
            or any(not np.isfinite(value) for value in (*density, *temperature))
            or density[0] <= 0.0
            or density[1] <= density[0]
            or temperature[0] <= 0.0
            or temperature[1] <= temperature[0]
            or not np.isfinite(inverse)
            or inverse < 0.0
            or not np.isfinite(derivative)
            or derivative < 0.0
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("EOS derivative certificate values are invalid.")
        passed = inverse <= tolerance_ and derivative <= tolerance_
        self.model_id = identifier
        self.density_bounds = (density[0], density[1])
        self.temperature_bounds = (temperature[0], temperature[1])
        self.maximum_inverse_residual = inverse
        self.maximum_derivative_residual = derivative
        self.tolerance = tolerance_
        self.passed = passed
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "eos-derivative-certificate",
                "model": identifier,
                "density_bounds": density,
                "temperature_bounds": temperature,
                "maximum_inverse_residual": inverse,
                "maximum_derivative_residual": derivative,
                "tolerance": tolerance_,
                "passed": passed,
            }
        )


class EOSConvexityCertificate(StrictModule, NonTrainableState):
    """Positive sound-speed and fundamental-derivative evidence for one EOS."""

    model_id: str = eqx.field(static=True)
    minimum_sound_speed_squared: float = eqx.field(static=True)
    minimum_fundamental_derivative: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        model_id: str,
        /,
        *,
        minimum_sound_speed_squared: float,
        minimum_fundamental_derivative: float,
        tolerance: float = 0.0,
    ):
        identifier = str(model_id)
        speed = float(minimum_sound_speed_squared)
        derivative = float(minimum_fundamental_derivative)
        tolerance_ = float(tolerance)
        if (
            not identifier
            or not np.isfinite(speed)
            or not np.isfinite(derivative)
            or not np.isfinite(tolerance_)
            or tolerance_ < 0.0
        ):
            raise ValueError("EOS convexity certificate values are invalid.")
        passed = speed > tolerance_ and derivative > tolerance_
        self.model_id = identifier
        self.minimum_sound_speed_squared = speed
        self.minimum_fundamental_derivative = derivative
        self.tolerance = tolerance_
        self.passed = passed
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "eos-convexity-certificate",
                "model": identifier,
                "minimum_sound_speed_squared": speed,
                "minimum_fundamental_derivative": derivative,
                "tolerance": tolerance_,
                "passed": passed,
            }
        )


class ThermallyPerfectGasMaterial(AbstractThermodynamicMaterial):
    """Ideal thermal EOS with a polynomial cp(T) and monotone caloric inversion."""

    cp_coefficients: tuple[float, ...] = eqx.field(static=True)
    gas_constant: float = eqx.field(static=True)
    reference_temperature: float = eqx.field(static=True)
    reference_internal_energy: float = eqx.field(static=True)
    temperature_bounds: tuple[float, float] = eqx.field(static=True)
    inversion_iterations: int = eqx.field(static=True)

    def __init__(
        self,
        cp_coefficients: Sequence[float],
        gas_constant: float,
        /,
        *,
        reference_temperature: float = 1.0,
        reference_internal_energy: float = 0.0,
        temperature_bounds: Sequence[float] = (1.0e-3, 1.0e4),
        inversion_iterations: int = 48,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-12,
    ):
        coefficients = tuple(float(value) for value in cp_coefficients)
        gas_constant_ = float(gas_constant)
        reference_temperature_ = float(reference_temperature)
        reference_energy = float(reference_internal_energy)
        bounds = tuple(float(value) for value in temperature_bounds)
        iterations = int(inversion_iterations)
        density_floor_ = float(density_floor)
        pressure_floor_ = float(pressure_floor)
        if (
            not coefficients
            or any(not np.isfinite(value) for value in coefficients)
            or not np.isfinite(gas_constant_)
            or gas_constant_ <= 0.0
            or not np.isfinite(reference_temperature_)
            or len(bounds) != 2
            or any(not np.isfinite(value) for value in bounds)
            or bounds[0] <= 0.0
            or bounds[1] <= bounds[0]
            or not bounds[0] <= reference_temperature_ <= bounds[1]
            or not np.isfinite(reference_energy)
            or iterations <= 0
            or not np.isfinite(density_floor_)
            or density_floor_ <= 0.0
            or not np.isfinite(pressure_floor_)
            or pressure_floor_ <= 0.0
        ):
            raise ValueError("Thermally-perfect gas parameters are invalid.")
        cv_polynomial = np.polynomial.Polynomial(
            (coefficients[0] - gas_constant_, *coefficients[1:])
        )
        candidates = [bounds[0], bounds[1]]
        for root in cv_polynomial.deriv().roots():
            if (
                abs(root.imag) <= 64.0 * np.finfo(float).eps
                and bounds[0] <= root.real <= bounds[1]
            ):
                candidates.append(float(root.real))
        if min(float(cv_polynomial(value)) for value in candidates) <= 0.0:
            raise ValueError("cp(T)-R must remain positive over temperature_bounds.")
        self.cp_coefficients = coefficients
        self.gas_constant = gas_constant_
        self.reference_temperature = reference_temperature_
        self.reference_internal_energy = reference_energy
        self.temperature_bounds = (bounds[0], bounds[1])
        self.inversion_iterations = iterations
        self.density_floor = density_floor_
        self.pressure_floor = pressure_floor_
        self.material_id = canonical_fingerprint(
            {
                "kind": "thermally-perfect-gas-material",
                "cp_coefficients": coefficients,
                "gas_constant": gas_constant_,
                "reference_temperature": reference_temperature_,
                "reference_internal_energy": reference_energy,
                "temperature_bounds": bounds,
                "inversion_iterations": iterations,
                "density_floor": density_floor_,
                "pressure_floor": pressure_floor_,
            }
        )

    def specific_heat_cp_from_temperature(self, temperature: ArrayLike, /) -> Array:
        value = jnp.asarray(temperature)
        result = jnp.zeros_like(value)
        for coefficient in reversed(self.cp_coefficients):
            result = result * value + coefficient
        return result

    def specific_internal_energy_from_temperature(
        self, temperature: ArrayLike, /
    ) -> Array:
        value = jnp.asarray(temperature)
        reference = jnp.asarray(self.reference_temperature, dtype=value.dtype)
        result = jnp.full_like(value, self.reference_internal_energy)
        for order, coefficient in enumerate(self.cp_coefficients):
            power = order + 1
            caloric_coefficient = coefficient - (self.gas_constant if order == 0 else 0.0)
            result = result + (caloric_coefficient / power) * (
                value**power - reference**power
            )
        return result

    def temperature_from_specific_internal_energy(
        self, specific_internal_energy: ArrayLike, /
    ) -> Array:
        target = jnp.asarray(specific_internal_energy)
        lower_bound = jnp.asarray(self.temperature_bounds[0], dtype=target.dtype)
        upper_bound = jnp.asarray(self.temperature_bounds[1], dtype=target.dtype)
        lower_energy = self.specific_internal_energy_from_temperature(lower_bound)
        upper_energy = self.specific_internal_energy_from_temperature(upper_bound)
        target = eqx.error_if(
            target,
            jnp.any(
                ~jnp.isfinite(target) | (target < lower_energy) | (target > upper_energy)
            ),
            "Specific internal energy lies outside the certified caloric range.",
        )
        fraction = (target - lower_energy) / (upper_energy - lower_energy)
        temperature = lower_bound + fraction * (upper_bound - lower_bound)
        lower = jnp.full_like(target, lower_bound)
        upper = jnp.full_like(target, upper_bound)

        def body(_, carry):
            current, lower_, upper_ = carry
            energy = self.specific_internal_energy_from_temperature(current)
            residual = energy - target
            lower_next = jnp.where(residual <= 0.0, current, lower_)
            upper_next = jnp.where(residual > 0.0, current, upper_)
            cv = self.specific_heat_cp_from_temperature(current) - self.gas_constant
            newton = current - residual / cv
            midpoint = 0.5 * (lower_next + upper_next)
            use_newton = (
                jnp.isfinite(newton) & (newton > lower_next) & (newton < upper_next)
            )
            return jnp.where(use_newton, newton, midpoint), lower_next, upper_next

        temperature, _, _ = jax.lax.fori_loop(
            0, self.inversion_iterations, body, (temperature, lower, upper)
        )
        return temperature

    def pressure(self, density: Array, specific_internal_energy: Array, /) -> Array:
        density_ = jnp.asarray(density)
        temperature = self.temperature_from_specific_internal_energy(
            specific_internal_energy
        )
        return density_ * self.gas_constant * temperature

    def specific_internal_energy(self, density: Array, pressure: Array, /) -> Array:
        temperature = self.temperature(density, pressure)
        return self.specific_internal_energy_from_temperature(temperature)

    def temperature(self, density: Array, pressure: Array, /) -> Array:
        density_ = jnp.asarray(density)
        pressure_ = jnp.asarray(pressure)
        temperature = pressure_ / (density_ * self.gas_constant)
        return eqx.error_if(
            temperature,
            jnp.any(
                ~jnp.isfinite(temperature)
                | (temperature < self.temperature_bounds[0])
                | (temperature > self.temperature_bounds[1])
            ),
            "Thermodynamic state lies outside temperature_bounds.",
        )

    def sound_speed(self, density: Array, pressure: Array, /) -> Array:
        temperature = self.temperature(density, pressure)
        cp = self.specific_heat_cp_from_temperature(temperature)
        gamma = cp / (cp - self.gas_constant)
        return jnp.sqrt(gamma * jnp.asarray(pressure) / jnp.asarray(density))

    def specific_enthalpy(self, density: Array, pressure: Array, /) -> Array:
        temperature = self.temperature(density, pressure)
        return self.specific_internal_energy_from_temperature(temperature) + (
            self.gas_constant * temperature
        )

    def specific_heat_cp(self, density: Array, pressure: Array, /) -> Array:
        return self.specific_heat_cp_from_temperature(self.temperature(density, pressure))

    def admissible(self, density: Array, pressure: Array, /) -> Array:
        density_ = jnp.asarray(density)
        pressure_ = jnp.asarray(pressure)
        temperature = pressure_ / (density_ * self.gas_constant)
        return (
            (density_ >= self.density_floor)
            & (pressure_ >= self.pressure_floor)
            & (temperature >= self.temperature_bounds[0])
            & (temperature <= self.temperature_bounds[1])
        )


class ResearchRealGasMaterial(AbstractThermodynamicMaterial):
    """Research EOS accepted only with exact derivative and convexity certificates."""

    pressure_provider: Callable = eqx.field(static=True)
    energy_provider: Callable = eqx.field(static=True)
    temperature_provider: Callable = eqx.field(static=True)
    sound_speed_provider: Callable = eqx.field(static=True)
    enthalpy_provider: Callable = eqx.field(static=True)
    heat_capacity_provider: Callable = eqx.field(static=True)
    derivative_certificate: EOSDerivativeCertificate
    convexity_certificate: EOSConvexityCertificate
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        model_id: str,
        /,
        *,
        pressure_provider: Callable,
        energy_provider: Callable,
        temperature_provider: Callable,
        sound_speed_provider: Callable,
        enthalpy_provider: Callable,
        heat_capacity_provider: Callable,
        derivative_certificate: EOSDerivativeCertificate | None = None,
        convexity_certificate: EOSConvexityCertificate | None = None,
        density_floor: float = 1.0e-12,
        pressure_floor: float = 1.0e-12,
    ):
        identifier = str(model_id)
        providers = (
            pressure_provider,
            energy_provider,
            temperature_provider,
            sound_speed_provider,
            enthalpy_provider,
            heat_capacity_provider,
        )
        density_floor_ = float(density_floor)
        pressure_floor_ = float(pressure_floor)
        if not identifier or any(not callable(provider) for provider in providers):
            raise TypeError(
                "Real-gas EOS providers must be callable and model_id nonempty."
            )
        if not isinstance(derivative_certificate, EOSDerivativeCertificate):
            raise ValueError(
                "Real-gas research use requires an EOS derivative certificate."
            )
        if not isinstance(convexity_certificate, EOSConvexityCertificate):
            raise ValueError(
                "Real-gas research use requires an EOS convexity certificate."
            )
        if (
            derivative_certificate.model_id != identifier
            or convexity_certificate.model_id != identifier
            or not derivative_certificate.passed
            or not convexity_certificate.passed
        ):
            raise ValueError(
                "Real-gas certificates must pass and target the exact model."
            )
        if (
            not np.isfinite(density_floor_)
            or density_floor_ <= 0.0
            or not np.isfinite(pressure_floor_)
            or pressure_floor_ <= 0.0
        ):
            raise ValueError("Real-gas floors must be finite and positive.")
        self.model_id = identifier
        self.pressure_provider = pressure_provider
        self.energy_provider = energy_provider
        self.temperature_provider = temperature_provider
        self.sound_speed_provider = sound_speed_provider
        self.enthalpy_provider = enthalpy_provider
        self.heat_capacity_provider = heat_capacity_provider
        self.derivative_certificate = derivative_certificate
        self.convexity_certificate = convexity_certificate
        self.density_floor = density_floor_
        self.pressure_floor = pressure_floor_
        self.material_id = canonical_fingerprint(
            {
                "kind": "research-real-gas-material",
                "model": identifier,
                "derivative_certificate": derivative_certificate.certificate_id,
                "convexity_certificate": convexity_certificate.certificate_id,
                "density_floor": density_floor_,
                "pressure_floor": pressure_floor_,
            }
        )

    def pressure(self, density: Array, specific_internal_energy: Array, /) -> Array:
        return jnp.asarray(self.pressure_provider(density, specific_internal_energy))

    def specific_internal_energy(self, density: Array, pressure: Array, /) -> Array:
        return jnp.asarray(self.energy_provider(density, pressure))

    def temperature(self, density: Array, pressure: Array, /) -> Array:
        return jnp.asarray(self.temperature_provider(density, pressure))

    def sound_speed(self, density: Array, pressure: Array, /) -> Array:
        return jnp.asarray(self.sound_speed_provider(density, pressure))

    def specific_enthalpy(self, density: Array, pressure: Array, /) -> Array:
        return jnp.asarray(self.enthalpy_provider(density, pressure))

    def specific_heat_cp(self, density: Array, pressure: Array, /) -> Array:
        return jnp.asarray(self.heat_capacity_provider(density, pressure))

    def admissible(self, density: Array, pressure: Array, /) -> Array:
        density_ = jnp.asarray(density)
        pressure_ = jnp.asarray(pressure)
        temperature = self.temperature(density_, pressure_)
        sound = self.sound_speed(density_, pressure_)
        heat_capacity = self.specific_heat_cp(density_, pressure_)
        density_bounds = self.derivative_certificate.density_bounds
        temperature_bounds = self.derivative_certificate.temperature_bounds
        return (
            (density_ >= jnp.maximum(self.density_floor, density_bounds[0]))
            & (density_ <= density_bounds[1])
            & (pressure_ >= self.pressure_floor)
            & jnp.isfinite(temperature)
            & (temperature >= temperature_bounds[0])
            & (temperature <= temperature_bounds[1])
            & jnp.isfinite(sound)
            & (sound > 0.0)
            & jnp.isfinite(heat_capacity)
            & (heat_capacity > 0.0)
        )


__all__ = [
    "EOSConvexityCertificate",
    "EOSDerivativeCertificate",
    "ResearchRealGasMaterial",
    "ThermallyPerfectGasMaterial",
]
