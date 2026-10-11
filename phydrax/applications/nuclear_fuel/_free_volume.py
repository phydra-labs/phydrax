#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Communicating free-gas volumes and the rod gas-pressure closure.

Every free-gas volume is counted by exactly one rule:

* plenum: cladding bore over the current plenum length, minus the spring;
* radial gap: annulus between current fuel and cladding radii at each
  fuel-interface site (assembled by the plenum response);
* dishes and chamfers: fabricated per-pellet voids per unit stack length;
* relocation cracks: the volume swept by the relocation eigenstrain of the
  pellet surface (never counted again as gap);
* open porosity: fabricated connected porosity, reduced by densification.

Closed intragranular and grain-face bubbles are gas storage, not free volume.
All volumes communicate at one pressure, a modeling assumption of this closure.
With piecewise-uniform temperatures, the ideal-gas pressure is
``p = n R / sum_j (V_j / T_j)``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._constants import MOLAR_GAS_CONSTANT_J_PER_MOL_K
from ._specification import FuelRodSpecification


if TYPE_CHECKING:
    from ._numerical_views import NumericalFreeVolumeModel


class FreeVolumeComponents(StrictModule):
    """Current free-gas volumes [m^3] and their gas temperatures [K]."""

    plenum_volume: Array
    plenum_temperature: Array
    gap_volume: Array
    gap_temperature: Array
    void_volume: Array
    void_temperature: Array

    @property
    def total_volume(self) -> Array:
        return (
            self.plenum_volume
            + jnp.sum(self.gap_volume, axis=-1)
            + jnp.sum(self.void_volume, axis=-1)
        )

    @property
    def volume_over_temperature(self) -> Array:
        return (
            self.plenum_volume / self.plenum_temperature
            + jnp.sum(self.gap_volume / self.gap_temperature, axis=-1)
            + jnp.sum(self.void_volume / self.void_temperature, axis=-1)
        )


class FreeVolumeModel(StrictModule, NonTrainableState):
    """Fabricated free-volume geometry of one rod.

    ``plenum_reference_length`` [m] and ``spring_volume`` [m^3] describe the
    as-fabricated plenum, ``dish_chamfer_volume_per_length`` [m^2] the pellet
    end voids per unit stack length, ``open_porosity_fraction`` the fabricated
    connected porosity, and ``fuel_radius`` [m] the reference pellet radius.
    """

    plenum_reference_length: float = eqx.field(static=True)
    spring_volume: float = eqx.field(static=True)
    dish_chamfer_volume_per_length: float = eqx.field(static=True)
    open_porosity_fraction: float = eqx.field(static=True)
    fuel_radius: float = eqx.field(static=True)

    def __init__(self, specification: FuelRodSpecification, /) -> None:
        if not isinstance(specification, FuelRodSpecification):
            raise TypeError("specification must be FuelRodSpecification.")
        pellet = specification.pellet
        self.plenum_reference_length = specification.rod.plenum_length
        self.spring_volume = specification.rod.spring_volume
        self.dish_chamfer_volume_per_length = (
            pellet.dish_volume + pellet.chamfer_volume
        ) / pellet.height
        self.open_porosity_fraction = pellet.open_porosity_fraction
        self.fuel_radius = pellet.radius
        plenum_bore_volume = (
            math.pi
            * specification.cladding.inner_radius**2
            * self.plenum_reference_length
        )
        if self.spring_volume >= plenum_bore_volume:
            raise ValueError("The plenum spring cannot fill the plenum bore.")


def plenum_and_void_volumes(
    model: FreeVolumeModel | NumericalFreeVolumeModel,
    /,
    *,
    layer_lengths: Array,
    plenum_length: Array,
    plenum_bore_radius: Array,
    relocation_strain: Array,
    densification_volume_fraction: Array,
) -> tuple[Array, Array]:
    """Plenum gas volume [m^3] and per-zone fuel void volume [m^3] of one rod.

    Per-zone arrays have trailing axis ``K`` (axial zones); lengths are in m.
    The void volume sums relocation cracks, open porosity, and dish/chamfer
    voids. ``densification_volume_fraction`` is the zone-mean fractional volume
    loss by densification (>= 0), which closes open porosity first.
    ``relocation_strain`` is the zone's linear relocation strain of the pellet
    surface.
    """

    cracks = (
        math.pi
        * model.fuel_radius**2
        * ((1.0 + relocation_strain) ** 2 - 1.0)
        * layer_lengths
    )
    open_pores = (
        math.pi
        * model.fuel_radius**2
        * jnp.maximum(model.open_porosity_fraction - densification_volume_fraction, 0.0)
        * layer_lengths
    )
    dishes = model.dish_chamfer_volume_per_length * layer_lengths
    plenum = math.pi * plenum_bore_radius**2 * plenum_length - model.spring_volume
    return plenum, cracks + open_pores + dishes


def gas_pressure_residual(
    pressure: Array, total_moles: Array, components: FreeVolumeComponents, /
) -> Array:
    """``p sum_j V_j/T_j - n R`` [J/K]; zero at the ideal-gas common pressure."""

    return (
        pressure * components.volume_over_temperature
        - total_moles * MOLAR_GAS_CONSTANT_J_PER_MOL_K
    )


__all__ = [
    "FreeVolumeComponents",
    "FreeVolumeModel",
    "plenum_and_void_volumes",
    "gas_pressure_residual",
]
