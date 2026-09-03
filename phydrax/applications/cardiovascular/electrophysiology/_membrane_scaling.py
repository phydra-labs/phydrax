#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Physical membrane scaling for cardiac reaction--conduction coupling."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array

from ._reaction import ArrayLike, require_positive_finite


@dataclass(frozen=True)
class CardiacMembraneScaling:
    """Surface/volume and capacitance scaling in the cardiovascular kernel units.

    Kernel base scales are mm, ms, mg, mV, kPa, and mm³.  Currents are
    outward-positive unless a method explicitly calls them applied/inward.

    The defaults correspond to chi = 1400 cm⁻¹ and Cm = 1 µF cm⁻².
    """

    membrane_surface_to_volume_per_mm: float = 140.0
    membrane_capacitance_uF_per_mm2: float = 0.01

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "membrane_surface_to_volume_per_mm",
            require_positive_finite(
                self.membrane_surface_to_volume_per_mm,
                "membrane_surface_to_volume_per_mm",
            ),
        )
        object.__setattr__(
            self,
            "membrane_capacitance_uF_per_mm2",
            require_positive_finite(
                self.membrane_capacitance_uF_per_mm2,
                "membrane_capacitance_uF_per_mm2",
            ),
        )

    @property
    def volumetric_capacitance_uF_per_mm3(self) -> float:
        """Return chi Cm, the tissue-volume capacitance density."""
        return (
            self.membrane_surface_to_volume_per_mm * self.membrane_capacitance_uF_per_mm2
        )

    @property
    def membrane_surface_to_volume_per_m(self) -> float:
        """Exact SI conversion of chi from mm⁻¹ to m⁻¹."""
        return self.membrane_surface_to_volume_per_mm * 1_000.0

    @property
    def membrane_capacitance_F_per_m2(self) -> float:
        """Exact SI conversion; 1 µF/mm² equals 1 F/m²."""
        return self.membrane_capacitance_uF_per_mm2

    @property
    def volumetric_capacitance_F_per_m3(self) -> float:
        return self.volumetric_capacitance_uF_per_mm3 * 1_000.0

    def conductivity_mS_per_mm_to_S_per_m(self, conductivity: ArrayLike, /) -> Array:
        """Convert conductivity exactly; the numeric factor is one."""
        return jnp.asarray(conductivity)

    def surface_current_uA_per_mm2_to_A_per_m2(self, current: ArrayLike, /) -> Array:
        """Convert surface current exactly; the numeric factor is one."""
        return jnp.asarray(current)

    def volume_current_uA_per_mm3_to_A_per_m3(self, current: ArrayLike, /) -> Array:
        """Convert volume current; 1 µA/mm³ equals 1000 A/m³."""
        return jnp.asarray(current) * 1_000.0

    def voltage_rate_mV_per_ms_to_V_per_s(self, rate: ArrayLike, /) -> Array:
        """Convert voltage rate exactly; the numeric factor is one."""
        return jnp.asarray(rate)

    def surface_current_to_volume_current(
        self,
        outward_current_uA_per_mm2: ArrayLike,
        /,
    ) -> Array:
        """Convert outward membrane current density to tissue-volume current."""
        return self.membrane_surface_to_volume_per_mm * jnp.asarray(
            outward_current_uA_per_mm2
        )

    def volume_current_to_surface_current(
        self,
        outward_current_uA_per_mm3: ArrayLike,
        /,
    ) -> Array:
        return (
            jnp.asarray(outward_current_uA_per_mm3)
            / self.membrane_surface_to_volume_per_mm
        )

    def outward_surface_current_to_voltage_rate(
        self,
        outward_current_uA_per_mm2: ArrayLike,
        /,
    ) -> Array:
        """Return the negative voltage rate produced by outward surface current."""
        return (
            -jnp.asarray(outward_current_uA_per_mm2)
            / self.membrane_capacitance_uF_per_mm2
        )

    def outward_volume_current_to_voltage_rate(
        self,
        outward_current_uA_per_mm3: ArrayLike,
        /,
    ) -> Array:
        """Return the negative voltage rate produced by outward volume current."""
        return (
            -jnp.asarray(outward_current_uA_per_mm3)
            / self.volumetric_capacitance_uF_per_mm3
        )

    def applied_volume_current_to_voltage_rate(
        self,
        inward_current_uA_per_mm3: ArrayLike,
        /,
    ) -> Array:
        """Return the positive voltage rate for an inward-positive applied stimulus."""
        return (
            jnp.asarray(inward_current_uA_per_mm3)
            / self.volumetric_capacitance_uF_per_mm3
        )

    def conductivity_to_diffusivity_mm2_per_ms(
        self,
        conductivity_mS_per_mm: ArrayLike,
        /,
    ) -> Array:
        """Map scalar/tensor conductivity to the monodomain diffusion coefficient."""
        conductivity = jnp.asarray(conductivity_mS_per_mm)
        return conductivity / self.volumetric_capacitance_uF_per_mm3

    def conductive_divergence_to_voltage_rate(
        self,
        divergence_uA_per_mm3: ArrayLike,
        /,
    ) -> Array:
        """Scale div(sigma grad V) into its positive monodomain voltage rate."""
        return jnp.asarray(divergence_uA_per_mm3) / self.volumetric_capacitance_uF_per_mm3


__all__ = [
    "CardiacMembraneScaling",
]
