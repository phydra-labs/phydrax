#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class AbstractBarotropicMaterial(StrictModule, NonTrainableState):
    """Pressure and barotropic energy closure depending only on density."""

    density_floor: float = eqx.field(static=True)
    material_id: str = eqx.field(static=True)

    @abc.abstractmethod
    def pressure(self, density: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def density_from_pressure(self, pressure: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def sound_speed(self, density: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def specific_internal_energy(self, density: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def admissible(self, density: Array, /) -> Array:
        raise NotImplementedError


class TaitBarotropicMaterial(AbstractBarotropicMaterial):
    """Tait barotrope parameterized by its reference acoustic speed."""

    reference_density: float = eqx.field(static=True)
    reference_sound_speed: float = eqx.field(static=True)
    exponent: float = eqx.field(static=True)
    background_pressure: float = eqx.field(static=True)
    stiffness: float = eqx.field(static=True)

    def __init__(
        self,
        reference_density: float,
        reference_sound_speed: float,
        /,
        *,
        exponent: float = 7.0,
        background_pressure: float = 0.0,
        density_floor: float = 1.0e-12,
    ):
        density = float(reference_density)
        sound_speed = float(reference_sound_speed)
        exponent_ = float(exponent)
        background = float(background_pressure)
        floor = float(density_floor)
        if (
            not np.isfinite(density)
            or density <= 0.0
            or not np.isfinite(sound_speed)
            or sound_speed <= 0.0
            or not np.isfinite(exponent_)
            or exponent_ <= 1.0
            or not np.isfinite(background)
            or not np.isfinite(floor)
            or floor <= 0.0
        ):
            raise ValueError(
                "Tait parameters require positive finite reference density, sound "
                "speed, exponent > 1, and density floor; background pressure must "
                "be finite."
            )
        stiffness = density * sound_speed**2 / exponent_
        self.reference_density = density
        self.reference_sound_speed = sound_speed
        self.exponent = exponent_
        self.background_pressure = background
        self.density_floor = floor
        self.stiffness = stiffness
        self.material_id = canonical_fingerprint(
            {
                "kind": "tait-barotropic-material",
                "reference_density": density,
                "reference_sound_speed": sound_speed,
                "exponent": exponent_,
                "background_pressure": background,
                "density_floor": floor,
                "stiffness": stiffness,
            }
        )

    def _ratio(self, density: Array, /) -> Array:
        density_ = jnp.asarray(density)
        if not jnp.issubdtype(density_.dtype, jnp.inexact):
            density_ = density_.astype(jnp.float32)
        return density_ / jnp.asarray(self.reference_density, dtype=density_.dtype)

    def pressure(self, density: Array, /) -> Array:
        ratio = self._ratio(density)
        return self.stiffness * (ratio**self.exponent - 1.0) + self.background_pressure

    def density_from_pressure(self, pressure: Array, /) -> Array:
        pressure_ = jnp.asarray(pressure)
        if not jnp.issubdtype(pressure_.dtype, jnp.inexact):
            pressure_ = pressure_.astype(jnp.float32)
        stiffness = jnp.asarray(self.stiffness, dtype=pressure_.dtype)
        background = jnp.asarray(self.background_pressure, dtype=pressure_.dtype)
        base = 1.0 + (pressure_ - background) / stiffness
        base = eqx.error_if(
            base,
            jnp.any(~jnp.isfinite(base) | (base <= 0.0)),
            "Tait pressure is outside the invertible density range.",
        )
        return self.reference_density * base ** (1.0 / self.exponent)

    def sound_speed(self, density: Array, /) -> Array:
        ratio = self._ratio(density)
        return self.reference_sound_speed * ratio ** (0.5 * (self.exponent - 1.0))

    def specific_internal_energy(self, density: Array, /) -> Array:
        ratio = self._ratio(density)
        reference = jnp.asarray(self.reference_density, dtype=ratio.dtype)
        stiffness = jnp.asarray(self.stiffness, dtype=ratio.dtype)
        background = jnp.asarray(self.background_pressure, dtype=ratio.dtype)
        exponent = jnp.asarray(self.exponent, dtype=ratio.dtype)
        compressive = (ratio ** (exponent - 1.0) - 1.0) / (exponent - 1.0)
        inverse_ratio_shift = 1.0 / ratio - 1.0
        return stiffness / reference * (
            compressive + inverse_ratio_shift
        ) + background / reference * (1.0 - 1.0 / ratio)

    def admissible(self, density: Array, /) -> Array:
        density_ = jnp.asarray(density)
        pressure = self.pressure(density_)
        sound_speed = self.sound_speed(density_)
        return (
            jnp.isfinite(density_)
            & (density_ >= self.density_floor)
            & jnp.isfinite(pressure)
            & jnp.isfinite(sound_speed)
            & (sound_speed > 0.0)
        )


class CavitationBarotropicBranch(IntEnum):
    VAPOR = 0
    TWO_PHASE = 1
    LIQUID = 2


class CavitationBarotropicState(StrictModule):
    pressure: Array
    sound_speed: Array
    vapor_fraction: Array
    specific_internal_energy: Array
    branch: Array
    lower_transition_margin: Array
    upper_transition_margin: Array
    finite: Array
    successful: Array
    material_id: str = eqx.field(static=True)


class HomogeneousEquilibriumCavitationMaterial(AbstractBarotropicMaterial):
    """Isothermal homogeneous-equilibrium liquid-vapour barotrope."""

    saturation_pressure: float = eqx.field(static=True)
    vapor_density: float = eqx.field(static=True)
    liquid_density: float = eqx.field(static=True)
    vapor_sound_speed: float = eqx.field(static=True)
    liquid_sound_speed: float = eqx.field(static=True)
    mixture_sound_speed: float = eqx.field(static=True)
    mixture_model: str = eqx.field(static=True)
    pressure_floor: float = eqx.field(static=True)
    lower_transition_pressure: float = eqx.field(static=True)
    upper_transition_pressure: float = eqx.field(static=True)
    wallis_a: float = eqx.field(static=True)
    wallis_b: float = eqx.field(static=True)
    liquid_transition_energy: float = eqx.field(static=True)

    def __init__(
        self,
        saturation_pressure: float,
        vapor_density: float,
        liquid_density: float,
        vapor_sound_speed: float,
        liquid_sound_speed: float,
        /,
        *,
        mixture_model: str = "wallis",
        mixture_sound_speed: float | None = None,
        pressure_floor: float = 0.0,
        density_floor: float = 1.0e-12,
    ):
        p_sat = float(saturation_pressure)
        rho_v = float(vapor_density)
        rho_l = float(liquid_density)
        c_v = float(vapor_sound_speed)
        c_l = float(liquid_sound_speed)
        floor = float(density_floor)
        p_floor = float(pressure_floor)
        model = str(mixture_model)
        if (
            any(
                not np.isfinite(value)
                for value in (p_sat, rho_v, rho_l, c_v, c_l, floor, p_floor)
            )
            or rho_v <= floor
            or rho_l <= rho_v
            or c_v <= 0.0
            or c_l <= 0.0
            or floor <= 0.0
            or model not in ("linear", "wallis")
        ):
            raise ValueError("Homogeneous-equilibrium cavitation parameters are invalid.")
        delta = rho_l - rho_v
        vapor_modulus = rho_v * c_v**2
        liquid_modulus = rho_l * c_l**2
        a = (rho_l / vapor_modulus - rho_v / liquid_modulus) / delta
        b = (1.0 / liquid_modulus - 1.0 / vapor_modulus) / delta
        midpoint = 0.5 * (rho_v + rho_l)
        midpoint_compressibility = a + b * midpoint
        if (
            not np.isfinite(a)
            or not np.isfinite(b)
            or a <= 0.0
            or midpoint_compressibility <= 0.0
            or a + b * rho_v <= 0.0
            or a + b * rho_l <= 0.0
        ):
            raise ValueError("Wallis mixture compressibility is not positive.")
        default_mixture_speed = float(
            np.sqrt(1.0 / (midpoint * midpoint_compressibility))
        )
        c_m = (
            default_mixture_speed
            if mixture_sound_speed is None
            else float(mixture_sound_speed)
        )
        if not np.isfinite(c_m) or c_m <= 0.0:
            raise ValueError("Mixture sound speed must be positive and finite.")
        if model == "linear":
            p_low = p_sat + c_m**2 * (rho_v - midpoint)
            p_high = p_sat + c_m**2 * (rho_l - midpoint)
            mixture_energy = self._linear_energy_increment_host(
                rho_v,
                rho_l,
                c_m**2,
                p_sat - c_m**2 * midpoint,
            )
        else:
            p_low = self._wallis_pressure_host(rho_v, midpoint, p_sat, a, b)
            p_high = self._wallis_pressure_host(rho_l, midpoint, p_sat, a, b)
            mixture_energy = self._wallis_energy_increment_host(
                rho_v,
                rho_l,
                midpoint,
                p_sat,
                a,
                b,
            )
        p_low = float(p_low)
        p_high = float(p_high)
        mixture_energy = float(mixture_energy)
        if (
            not np.isfinite(p_low)
            or not np.isfinite(p_high)
            or p_low >= p_high
            or not np.isfinite(mixture_energy)
        ):
            raise ValueError("Cavitation transition pressure/energy is invalid.")
        self.saturation_pressure = p_sat
        self.vapor_density = rho_v
        self.liquid_density = rho_l
        self.vapor_sound_speed = c_v
        self.liquid_sound_speed = c_l
        self.mixture_sound_speed = c_m
        self.mixture_model = model
        self.pressure_floor = p_floor
        self.density_floor = floor
        self.lower_transition_pressure = p_low
        self.upper_transition_pressure = p_high
        self.wallis_a = a
        self.wallis_b = b
        self.liquid_transition_energy = mixture_energy
        self.material_id = canonical_fingerprint(
            {
                "kind": "homogeneous-equilibrium-cavitation",
                "saturation_pressure": p_sat,
                "vapor_density": rho_v,
                "liquid_density": rho_l,
                "vapor_sound_speed": c_v,
                "liquid_sound_speed": c_l,
                "mixture_model": model,
                "mixture_sound_speed": c_m,
                "pressure_floor": p_floor,
                "density_floor": floor,
            }
        )

    @staticmethod
    def _linear_energy_increment_host(rho0, rho1, slope, intercept):
        return slope * np.log(rho1 / rho0) - intercept * (1.0 / rho1 - 1.0 / rho0)

    @staticmethod
    def _wallis_pressure_host(rho, reference, pressure, a, b):
        return (
            pressure + np.log(rho * (a + b * reference) / (reference * (a + b * rho))) / a
        )

    @staticmethod
    def _wallis_energy_primitive_host(rho, reference, pressure, a, b):
        log_ratio = np.log(rho / reference)
        log_compressibility = np.log((a + b * rho) / (a + b * reference))
        return (
            -pressure / rho
            + (
                (log_compressibility - log_ratio - 1.0) / rho
                - (b / a) * (log_ratio - log_compressibility)
            )
            / a
        )

    @classmethod
    def _wallis_energy_increment_host(cls, rho0, rho1, reference, pressure, a, b):
        return cls._wallis_energy_primitive_host(
            rho1, reference, pressure, a, b
        ) - cls._wallis_energy_primitive_host(rho0, reference, pressure, a, b)

    def _as_density(self, density: Array, /) -> Array:
        value = jnp.asarray(density)
        return (
            value.astype(jnp.float32)
            if not jnp.issubdtype(value.dtype, jnp.inexact)
            else value
        )

    def _mixture_density(self, density: Array, /) -> Array:
        return jnp.clip(density, self.vapor_density, self.liquid_density)

    def _linear_pressure(self, density: Array, /) -> Array:
        midpoint = 0.5 * (self.vapor_density + self.liquid_density)
        return self.saturation_pressure + self.mixture_sound_speed**2 * (
            density - midpoint
        )

    def _wallis_pressure(self, density: Array, /) -> Array:
        midpoint = 0.5 * (self.vapor_density + self.liquid_density)
        return (
            self.saturation_pressure
            + jnp.log(
                density
                * (self.wallis_a + self.wallis_b * midpoint)
                / (midpoint * (self.wallis_a + self.wallis_b * density))
            )
            / self.wallis_a
        )

    def pressure(self, density: Array, /) -> Array:
        value = self._as_density(density)
        mixture_density = self._mixture_density(value)
        mixture = (
            self._linear_pressure(mixture_density)
            if self.mixture_model == "linear"
            else self._wallis_pressure(mixture_density)
        )
        vapor = self.lower_transition_pressure + self.vapor_sound_speed**2 * (
            value - self.vapor_density
        )
        liquid = self.upper_transition_pressure + self.liquid_sound_speed**2 * (
            value - self.liquid_density
        )
        return jnp.where(
            value < self.vapor_density,
            vapor,
            jnp.where(value > self.liquid_density, liquid, mixture),
        )

    def density_from_pressure(self, pressure: Array, /) -> Array:
        value = jnp.asarray(pressure)
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(jnp.float32)
        vapor = (
            self.vapor_density
            + (value - self.lower_transition_pressure) / self.vapor_sound_speed**2
        )
        liquid = (
            self.liquid_density
            + (value - self.upper_transition_pressure) / self.liquid_sound_speed**2
        )
        mixture_pressure = jnp.clip(
            value,
            self.lower_transition_pressure,
            self.upper_transition_pressure,
        )
        if self.mixture_model == "linear":
            midpoint = 0.5 * (self.vapor_density + self.liquid_density)
            mixture = (
                midpoint
                + (mixture_pressure - self.saturation_pressure)
                / self.mixture_sound_speed**2
            )
        else:
            midpoint = 0.5 * (self.vapor_density + self.liquid_density)
            reference_ratio = midpoint / (self.wallis_a + self.wallis_b * midpoint)
            target = reference_ratio * jnp.exp(
                self.wallis_a * (mixture_pressure - self.saturation_pressure)
            )
            mixture = target * self.wallis_a / (1.0 - target * self.wallis_b)
        density = jnp.where(
            value < self.lower_transition_pressure,
            vapor,
            jnp.where(value > self.upper_transition_pressure, liquid, mixture),
        )
        return eqx.error_if(
            density,
            jnp.any(~jnp.isfinite(density) | (density < self.density_floor)),
            "Cavitation pressure is outside the invertible density range.",
        )

    def sound_speed(self, density: Array, /) -> Array:
        value = self._as_density(density)
        mixture_density = self._mixture_density(value)
        mixture = (
            jnp.full_like(value, self.mixture_sound_speed)
            if self.mixture_model == "linear"
            else jnp.sqrt(
                1.0
                / (mixture_density * (self.wallis_a + self.wallis_b * mixture_density))
            )
        )
        return jnp.where(
            value < self.vapor_density,
            self.vapor_sound_speed,
            jnp.where(value > self.liquid_density, self.liquid_sound_speed, mixture),
        )

    @staticmethod
    def _linear_energy_increment(
        rho0: Array, rho1: Array, slope: Array, intercept: Array, /
    ) -> Array:
        return slope * jnp.log(rho1 / rho0) - intercept * (1.0 / rho1 - 1.0 / rho0)

    def _mixture_energy(self, density: Array, /) -> Array:
        rho_v = jnp.asarray(self.vapor_density, dtype=density.dtype)
        if self.mixture_model == "linear":
            midpoint = 0.5 * (self.vapor_density + self.liquid_density)
            slope = jnp.asarray(self.mixture_sound_speed**2, dtype=density.dtype)
            intercept = jnp.asarray(
                self.saturation_pressure - self.mixture_sound_speed**2 * midpoint,
                dtype=density.dtype,
            )
            return self._linear_energy_increment(rho_v, density, slope, intercept)
        midpoint = jnp.asarray(
            0.5 * (self.vapor_density + self.liquid_density), dtype=density.dtype
        )
        a = jnp.asarray(self.wallis_a, dtype=density.dtype)
        b = jnp.asarray(self.wallis_b, dtype=density.dtype)
        pressure = jnp.asarray(self.saturation_pressure, dtype=density.dtype)

        def primitive(rho):
            log_ratio = jnp.log(rho / midpoint)
            log_compressibility = jnp.log((a + b * rho) / (a + b * midpoint))
            return (
                -pressure / rho
                + (
                    (log_compressibility - log_ratio - 1.0) / rho
                    - (b / a) * (log_ratio - log_compressibility)
                )
                / a
            )

        return primitive(density) - primitive(rho_v)

    def specific_internal_energy(self, density: Array, /) -> Array:
        value = self._as_density(density)
        rho_v = jnp.asarray(self.vapor_density, dtype=value.dtype)
        rho_l = jnp.asarray(self.liquid_density, dtype=value.dtype)
        vapor_intercept = jnp.asarray(
            self.lower_transition_pressure
            - self.vapor_sound_speed**2 * self.vapor_density,
            dtype=value.dtype,
        )
        vapor = self._linear_energy_increment(
            rho_v,
            value,
            jnp.asarray(self.vapor_sound_speed**2, dtype=value.dtype),
            vapor_intercept,
        )
        mixture = self._mixture_energy(self._mixture_density(value))
        liquid_intercept = jnp.asarray(
            self.upper_transition_pressure
            - self.liquid_sound_speed**2 * self.liquid_density,
            dtype=value.dtype,
        )
        liquid = jnp.asarray(
            self.liquid_transition_energy, dtype=value.dtype
        ) + self._linear_energy_increment(
            rho_l,
            value,
            jnp.asarray(self.liquid_sound_speed**2, dtype=value.dtype),
            liquid_intercept,
        )
        return jnp.where(
            value < rho_v,
            vapor,
            jnp.where(value > rho_l, liquid, mixture),
        )

    def vapor_fraction(self, density: Array, /) -> Array:
        value = self._as_density(density)
        return jnp.clip(
            (self.liquid_density - value) / (self.liquid_density - self.vapor_density),
            0.0,
            1.0,
        )

    def evaluate(self, density: Array, /) -> CavitationBarotropicState:
        value = self._as_density(density)
        pressure = self.pressure(value)
        sound = self.sound_speed(value)
        vapor_fraction = self.vapor_fraction(value)
        energy = self.specific_internal_energy(value)
        branch = jnp.where(
            value < self.vapor_density,
            int(CavitationBarotropicBranch.VAPOR),
            jnp.where(
                value > self.liquid_density,
                int(CavitationBarotropicBranch.LIQUID),
                int(CavitationBarotropicBranch.TWO_PHASE),
            ),
        ).astype(jnp.int32)
        finite = (
            jnp.isfinite(value)
            & jnp.isfinite(pressure)
            & jnp.isfinite(sound)
            & jnp.isfinite(energy)
        )
        successful = (
            finite
            & (value >= self.density_floor)
            & (pressure >= self.pressure_floor)
            & (sound > 0.0)
        )
        return CavitationBarotropicState(
            pressure,
            sound,
            vapor_fraction,
            energy,
            branch,
            value - self.vapor_density,
            self.liquid_density - value,
            finite,
            successful,
            self.material_id,
        )

    def admissible(self, density: Array, /) -> Array:
        return self.evaluate(density).successful


__all__ = [
    "AbstractBarotropicMaterial",
    "CavitationBarotropicBranch",
    "CavitationBarotropicState",
    "HomogeneousEquilibriumCavitationMaterial",
    "TaitBarotropicMaterial",
]
