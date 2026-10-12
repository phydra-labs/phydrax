"""Cumulative rod gas inventory and energy ledgers."""

from __future__ import annotations

from jax import Array

from ..._strict import StrictModule


class FuelGasLedger(StrictModule):
    """Gas moles in ``FUEL_GAS_SPECIES`` order (He, Ar, Kr, Xe, N2).

    ``balance_error`` is produced minus retained, released and exported moles;
    an intact rod exports none.
    """

    fill_moles: Array
    produced_moles: Array
    retained_moles: Array
    released_moles: Array
    exported_moles: Array
    balance_error: Array


class FuelEnergyLedger(StrictModule):
    """Cumulative energy [J] relative to the cold fabricated enthalpy.

    ``balance_error`` is stored plus delivered minus deposited energy.
    """

    deposited_fuel: Array
    deposited_cladding: Array
    deposited_direct_coolant: Array
    stored_fuel: Array
    stored_cladding: Array
    delivered_coolant: Array
    balance_error: Array
