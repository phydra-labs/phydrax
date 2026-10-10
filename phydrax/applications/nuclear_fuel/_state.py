"""Rod state: physical fields, material-site histories, and conserved ledgers."""

from __future__ import annotations

import equinox as eqx
from jax import Array

from ..._strict import StrictModule
from ...nuclear import ExposureState
from ._inventory import FuelEnergyLedger, FuelGasLedger
from ._radial_power import RadialPowerState
from ._site_state import CladdingSiteState, FuelSiteState


class FuelRodFields(StrictModule):
    """Physical fields in SI units, not scaled solver unknowns.

    Field order defines the solver coordinate layout. ``coolant_heat_flux``
    [W/m^2] is one value per layer of a layered rod and nodal on the RZ outer
    surface; its quadrature values come from the prepared geometry.
    """

    temperature_fuel: Array
    temperature_cladding: Array
    displacement_fuel: Array
    displacement_cladding: Array
    axial_strain_fuel: Array
    axial_strain_cladding: Array
    contact_pressure: Array
    friction_force: Array
    pressure: Array
    coolant_heat_flux: Array
    free_moles: Array
    channel_state: Array


class FuelRodState(StrictModule):
    """Accepted or candidate rod state at ``time`` on one prepared rod."""

    time: Array
    fields: FuelRodFields
    fuel_sites: FuelSiteState
    cladding_sites: CladdingSiteState
    exposure: ExposureState
    radial_power: RadialPowerState
    gas: FuelGasLedger
    energy: FuelEnergyLedger
    prepared_id: str = eqx.field(static=True)
