#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

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


__all__ = ["AbstractBarotropicMaterial", "TaitBarotropicMaterial"]
