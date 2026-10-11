#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One-site fuel gas block: total gas, helium and passive Kr/Xe.

``advance_fuel_gas`` evaluates the three models for one site.
"""

from __future__ import annotations

from typing import final, Literal

import jax.numpy as jnp

from ..._strict import StrictModule
from ...typing import (
    checked,
    Float64,
    Scalar,
)
from ._fission_gas import FissionGasModel
from ._gas_state import (
    FissionGasSiteState,
    FissionGasSpeciesState,
    FissionGasSpeciesStepResult,
    FissionGasStepResult,
    SiteDims,
)
from ._helium import HeliumGasModel, HeliumGasSiteState, HeliumGasStepResult
from ._identity import FUEL_GAS_SPECIES


# Slots of the five-species gas axis (FUEL_GAS_SPECIES order) of the gas block.
_HELIUM_SLOT = FUEL_GAS_SPECIES.index("He")
_KRYPTON_SLOT = FUEL_GAS_SPECIES.index("Kr")
_XENON_SLOT = FUEL_GAS_SPECIES.index("Xe")


@final
class FuelGasBlockResult(StrictModule):
    """Total-gas, helium and passive-species step results of one site.

    ``retained_species`` and ``released_species`` sum them on the
    five-species gas axis in FUEL_GAS_SPECIES order.
    """

    __strict_contract__ = True

    gas: FissionGasStepResult
    helium: HeliumGasStepResult
    species: FissionGasSpeciesStepResult
    retained_species: Float64[SiteDims, Literal[5]]
    released_species: Float64[SiteDims, Literal[5]]


@checked
def advance_fuel_gas(
    fission_gas: FissionGasModel,
    helium: HeliumGasModel,
    accepted_gas: FissionGasSiteState,
    accepted_helium: HeliumGasSiteState,
    accepted_species: FissionGasSpeciesState,
    /,
    *,
    temperature: Float64[Scalar],
    previous_temperature: Float64[Scalar],
    burnup_mwd_per_kghm: Float64[Scalar],
    fission_rate_density: Float64[Scalar],
    species_production_rate: Float64[Literal[5]],
    hydrostatic_stress: Float64[Scalar],
    time_step: Float64[Scalar],
    boundary_helium_pressure: Float64[Scalar],
) -> FuelGasBlockResult:
    """Advance one site's accepted histories at a supplied endpoint guess.

    Every evaluation starts at the same accepted time and uses the same step
    length. The caller owns batching and the thermomechanical fixed point;
    this block does not solve a coupled hydrostatic root. A single grain-law
    evaluation supplies the common candidate radius to both gas models.
    Species slots 0, 2 and 3 carry He, Kr and Xe respectively; the other
    two slots have no fuel-owned production or retention. Helium boundary
    transfer remains signed, including uptake from the rod free volume.
    """

    if accepted_gas.grain_radius.shape != () or accepted_helium.grain_radius.shape != ():
        raise ValueError("advance_fuel_gas requires per-site accepted gas histories.")

    grain_radius = fission_gas.grain_growth.advance_radius(
        accepted_gas.grain_radius,
        temperature,
        time_step,
        burnup_mwd_per_kghm=burnup_mwd_per_kghm,
    )
    gas = fission_gas.advance(
        accepted_gas,
        temperature=temperature,
        grain_radius=grain_radius,
        fission_rate_density=fission_rate_density,
        gas_production_rate=species_production_rate[_KRYPTON_SLOT]
        + species_production_rate[_XENON_SLOT],
        burnup_mwd_per_kghm=burnup_mwd_per_kghm,
        hydrostatic_stress=hydrostatic_stress,
        time_step=time_step,
        previous_temperature=previous_temperature,
    )
    helium_result = helium.advance(
        accepted_helium,
        temperature=temperature,
        fission_rate_density=fission_rate_density,
        gas_production_rate=species_production_rate[_HELIUM_SLOT],
        time_step=time_step,
        grain_radius=grain_radius,
        bubble_radius=fission_gas.trapping.bubble_radius(temperature),
        bubble_number_density=fission_gas.trapping.bubble_density(temperature),
        boundary_helium_pressure=boundary_helium_pressure,
    )
    species = fission_gas.advance_species(
        accepted_species,
        accepted_gas,
        gas,
        # The contiguous Kr, Xe slots in FISSION_GAS_TRACER_SPECIES order.
        species_production_rate=species_production_rate[_KRYPTON_SLOT : _XENON_SLOT + 1],
        time_step=time_step,
    )
    # Slots He, Ar, Kr, Xe, N2; argon and nitrogen have no fuel source.
    zero = jnp.zeros_like(helium_result.retained_total)
    retained_species = jnp.stack(
        (
            helium_result.retained_total,
            zero,
            species.retained_total[0],
            species.retained_total[1],
            zero,
        ),
    )
    released_species = jnp.stack(
        (
            helium_result.released_increment,
            zero,
            species.released_increment[0],
            species.released_increment[1],
            zero,
        ),
    )
    return FuelGasBlockResult(
        gas=gas,
        helium=helium_result,
        species=species,
        retained_species=retained_species,
        released_species=released_species,
    )


__all__ = [
    "FuelGasBlockResult",
    "advance_fuel_gas",
]
