#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Heavy-metal reference inventory and accumulated fuel exposure.

Burnup, FIMA, and fission density are referred to the as-fabricated heavy-metal
inventory and reference fuel volume. Densification, swelling, and relocation
change the current fuel volume but never these denominators.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._activation import AVOGADRO_PER_MOL
from ._identity import NuclideKey


# Atomic masses [u] of U-234, U-235, U-236, U-238: AME2020, W.J. Huang et al.,
# Chin. Phys. C 45, 030002 and M. Wang et al., Chin. Phys. C 45, 030003 (2021),
# file mass_1.mas20 (atomic-mass column).
_URANIUM_ISOTOPES = (
    NuclideKey(92, 234),
    NuclideKey(92, 235),
    NuclideKey(92, 236),
    NuclideKey(92, 238),
)
_URANIUM_ATOMIC_MASSES_U = np.asarray(
    [234.040950296, 235.043928117, 236.045566130, 238.050786936], dtype=np.float64
)
_URANIUM_ATOMIC_MASSES_U.setflags(write=False)
# Conventional standard atomic weight of oxygen: CIAAW, T. Prohaska et al.,
# "Standard atomic weights of the elements 2021", Pure Appl. Chem. 94, 573 (2022).
_OXYGEN_ATOMIC_WEIGHT = 15.999
# UO3 is the highest uranium oxide.
_MAXIMUM_OXYGEN_TO_METAL_RATIO = 3.0
_KILOGRAMS_PER_GRAM = 1.0e-3
# 1 MWd = 1e6 W * 86400 s.
_JOULES_PER_MEGAWATT_DAY = 8.64e10


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _weight_fraction(value: float, name: str, /) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a weight fraction in [0, 1].")
    return number


class HeavyMetalReference(StrictModule, NonTrainableState):
    """As-fabricated heavy-metal inventory per unit reference fuel volume.

    ``uranium_number_densities`` [atoms/m^3] follow ``uranium_isotopes``
    (U-234, U-235, U-236, U-238). ``heavy_metal_atom_density`` [atoms/m^3] and
    ``heavy_metal_mass_density`` [kg HM/m^3] are their sum and AME2020 mass
    sum.
    """

    uranium_number_densities: Array
    heavy_metal_atom_density: Array
    heavy_metal_mass_density: Array
    uranium_isotopes: tuple[NuclideKey, ...] = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(self, /, *, uranium_number_densities: ArrayLike) -> None:
        densities = np.array(uranium_number_densities, dtype=np.float64, copy=True)
        if densities.shape != (len(_URANIUM_ISOTOPES),):
            raise ValueError(
                "uranium_number_densities must list U-234, U-235, U-236, and U-238."
            )
        if not np.all(np.isfinite(densities)) or np.any(densities < 0.0):
            raise ValueError("uranium_number_densities must be finite and nonnegative.")
        atoms = float(np.sum(densities))
        if atoms <= 0.0:
            raise ValueError("The heavy-metal inventory must be positive.")
        mass = float(
            np.sum(densities * _URANIUM_ATOMIC_MASSES_U)
            * _KILOGRAMS_PER_GRAM
            / AVOGADRO_PER_MOL
        )
        self.uranium_number_densities = jnp.asarray(densities)
        self.heavy_metal_atom_density = jnp.asarray(atoms, dtype=jnp.float64)
        self.heavy_metal_mass_density = jnp.asarray(mass, dtype=jnp.float64)
        self.uranium_isotopes = _URANIUM_ISOTOPES
        self.reference_id = canonical_fingerprint(
            {
                "kind": "heavy-metal-reference",
                "uranium_isotopes": [value.nuclide_id for value in _URANIUM_ISOTOPES],
                "uranium_number_densities": array_tree_fingerprint(densities),
                "atomic_masses": "AME2020",
            }
        )

    @classmethod
    def oxide_fuel(
        cls,
        *,
        fuel_density: float,
        uranium_235_weight_fraction: float,
        oxygen_to_metal_ratio: float = 2.0,
        uranium_234_weight_fraction: float = 0.0,
        uranium_236_weight_fraction: float = 0.0,
    ) -> HeavyMetalReference:
        """Uranium-oxide fuel of mass density ``fuel_density`` [kg/m^3].

        Weight fractions are of uranium; U-238 is the remainder. The oxide
        formula unit is ``U O_x`` with ``x = oxygen_to_metal_ratio``.
        """

        density = _positive(fuel_density, "fuel_density")
        u234 = _weight_fraction(
            uranium_234_weight_fraction, "uranium_234_weight_fraction"
        )
        u235 = _weight_fraction(
            uranium_235_weight_fraction, "uranium_235_weight_fraction"
        )
        u236 = _weight_fraction(
            uranium_236_weight_fraction, "uranium_236_weight_fraction"
        )
        u238 = 1.0 - (u234 + u235 + u236)
        if u238 < 0.0:
            raise ValueError("Uranium isotope weight fractions must not exceed one.")
        ratio = _positive(oxygen_to_metal_ratio, "oxygen_to_metal_ratio")
        if ratio > _MAXIMUM_OXYGEN_TO_METAL_RATIO:
            raise ValueError("oxygen_to_metal_ratio cannot exceed 3 (UO3).")
        moles_per_gram = np.asarray([u234, u235, u236, u238]) / _URANIUM_ATOMIC_MASSES_U
        uranium_molar_mass = 1.0 / float(np.sum(moles_per_gram))
        oxide_molar_mass = uranium_molar_mass + ratio * _OXYGEN_ATOMIC_WEIGHT
        heavy_metal_atoms = (
            density / (oxide_molar_mass * _KILOGRAMS_PER_GRAM) * AVOGADRO_PER_MOL
        )
        atom_fractions = moles_per_gram * uranium_molar_mass
        return cls(uranium_number_densities=heavy_metal_atoms * atom_fractions)


class ExposureState(StrictModule):
    """Accumulated exposure of reference fuel volume.

    ``energy_per_initial_heavy_metal_mass`` [J/kg HM],
    ``fissions_per_initial_heavy_metal_atom`` (FIMA) [-], ``fission_density``
    [fissions/m^3 reference fuel], and ``fast_fluence`` [n/m^2, E > 1 MeV].
    """

    energy_per_initial_heavy_metal_mass: Array
    fissions_per_initial_heavy_metal_atom: Array
    fission_density: Array
    fast_fluence: Array

    @classmethod
    def zeros(cls, shape: tuple[int, ...] = (), /) -> ExposureState:
        """Fresh (unirradiated) fuel on ``shape`` sites."""

        zero = jnp.zeros(shape, dtype=jnp.float64)
        return cls(zero, zero, zero, zero)

    @property
    def burnup_mwd_per_kghm(self) -> Array:
        """Burnup [MWd/kg initial heavy metal]."""

        return self.energy_per_initial_heavy_metal_mass / _JOULES_PER_MEGAWATT_DAY


def advance_exposure(
    state: ExposureState,
    reference: HeavyMetalReference,
    /,
    *,
    fission_rate_density: ArrayLike,
    energy_per_fission: ArrayLike,
    fast_flux: ArrayLike,
    time_step: ArrayLike,
) -> ExposureState:
    """Accumulate one step of constant rates.

    ``fission_rate_density`` [fissions/m^3/s] is per unit reference fuel
    volume, ``energy_per_fission`` [J] is the recoverable energy per fission,
    ``fast_flux`` [n/m^2/s, E > 1 MeV], and ``time_step`` [s].
    """

    if not isinstance(state, ExposureState):
        raise TypeError("state must be ExposureState.")
    if not isinstance(reference, HeavyMetalReference):
        raise TypeError("reference must be HeavyMetalReference.")
    duration = jnp.asarray(time_step)
    fissions = jnp.asarray(fission_rate_density) * duration
    return ExposureState(
        state.energy_per_initial_heavy_metal_mass
        + fissions * jnp.asarray(energy_per_fission) / reference.heavy_metal_mass_density,
        state.fissions_per_initial_heavy_metal_atom
        + fissions / reference.heavy_metal_atom_density,
        state.fission_density + fissions,
        state.fast_fluence + jnp.asarray(fast_flux) * duration,
    )


__all__ = ["advance_exposure", "ExposureState", "HeavyMetalReference"]
