#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pressure acoustics, monopole radiation, impedance, and vibroacoustic power."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..qualification import CapabilityProfile, SupportTuple


@dataclass(frozen=True, slots=True)
class AcousticMedium:
    density_kg_m3: float
    sound_speed_m_s: float
    attenuation_np_m: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.density_kg_m3 <= 0.0
            or self.sound_speed_m_s <= 0.0
            or self.attenuation_np_m < 0.0
        ):
            raise ValueError("Acoustic medium properties are outside physical bounds.")

    @property
    def impedance_pa_s_m(self) -> float:
        return self.density_kg_m3 * self.sound_speed_m_s

    def wavenumber(self, frequency_hz: ArrayLike, /) -> Array:
        return (
            2.0 * jnp.pi * jnp.asarray(frequency_hz) / self.sound_speed_m_s
            - 1j * self.attenuation_np_m
        )


def monopole_pressure(
    source_strength_m3_s: ArrayLike,
    distance_m: ArrayLike,
    frequency_hz: ArrayLike,
    medium: AcousticMedium,
    /,
) -> Array:
    distance = jnp.maximum(jnp.asarray(distance_m), 1.0e-30)
    omega = 2.0 * jnp.pi * jnp.asarray(frequency_hz)
    return (
        1j
        * omega
        * medium.density_kg_m3
        * jnp.asarray(source_strength_m3_s)
        * jnp.exp(-1j * medium.wavenumber(frequency_hz) * distance)
        / (4.0 * jnp.pi * distance)
    )


def normal_incidence_transmission_loss(
    impedance_left: ArrayLike, impedance_right: ArrayLike, /
) -> Array:
    left = jnp.asarray(impedance_left)
    right = jnp.asarray(impedance_right)
    power_transmission = (
        4.0 * jnp.real(left) * jnp.real(right) / jnp.abs(left + right) ** 2
    )
    return -10.0 * jnp.log10(jnp.maximum(power_transmission, 1.0e-30))


def vibroacoustic_power(
    pressure_pa: ArrayLike, normal_velocity_m_s: ArrayLike, area_weights_m2: ArrayLike, /
) -> Array:
    return 0.5 * jnp.real(
        jnp.sum(
            jnp.asarray(pressure_pa)
            * jnp.conj(jnp.asarray(normal_velocity_m_s))
            * jnp.asarray(area_weights_m2)
        )
    )


def acoustics_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        ("acoustics.pressure-monopole", "three-dimensional-helmholtz-green"),
        ("acoustics.impedance-interface", "normal-incidence"),
        ("acoustics.vibroacoustic-power", "complex-boundary-work"),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, {"formulation": formulation}),),
            required_gates=("analytic-control", "energy-flux", "frequency-convention"),
        )
        for name, formulation in specs
    )


__all__ = [
    "AcousticMedium",
    "acoustics_candidate_profiles",
    "monopole_pressure",
    "normal_incidence_transmission_loss",
    "vibroacoustic_power",
]
