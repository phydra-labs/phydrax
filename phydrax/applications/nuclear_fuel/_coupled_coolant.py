#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative coolant-channel energy balance coupled to a fuel rod."""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...materials import iapws97_liquid_enthalpy, iapws97_liquid_state
from ...qualification import (
    CapabilityProfile,
    ReleaseIndex,
    ReleaseTrustPolicy,
    require_profile,
    SupportTuple,
)
from ..thermofluids import LiquidChannelHeatBalance
from ._coolant import SingleChannelCoolant
from ._irradiation import IrradiationConditions, IrradiationHistory


@final
class CoupledChannelCoolant(StrictModule):
    """A qualified channel's enthalpy unknowns coupled to the rod's heat rows.

    The channel state is one specific enthalpy [J/kg] per history zone, ordered
    inlet to outlet. Wall exchange is supplied as heat flux [W/m^2] multiplied
    by the current FE interface area [m^2]; both signs of exchange are kept.
    ``direct_power`` is channel heating [W] deposited directly in the coolant.
    The flow is prescribed by the history and must be positive at every
    breakpoint; there is no momentum or boiling model.
    """

    history: IrradiationHistory
    wall_transfer: SingleChannelCoolant
    balance: LiquidChannelHeatBalance
    initial_state: Array
    qualification_profile: CapabilityProfile = eqx.field(static=True)
    participant_id: str = eqx.field(static=True)

    def __init__(
        self,
        history: IrradiationHistory,
        wall_transfer: SingleChannelCoolant,
        release_index: ReleaseIndex,
        trust_policy: ReleaseTrustPolicy,
        /,
        *,
        profile_id: str,
        at_time: int,
    ) -> None:
        if not isinstance(history, IrradiationHistory):
            raise TypeError("history must be IrradiationHistory.")
        if not isinstance(wall_transfer, SingleChannelCoolant):
            raise TypeError("wall_transfer must be SingleChannelCoolant.")
        if np.any(np.asarray(history.coolant_mass_flux) <= 0.0):
            raise ValueError(
                "A coupled channel requires a positive coolant_mass_flux at every "
                "history time."
            )
        support = self.support_tuple(history, wall_transfer)
        qualification_profile = require_profile(
            release_index, profile_id, support, trust_policy, at_time=at_time
        )
        condition = history.conditions(0.0)
        water = iapws97_liquid_state(
            condition.coolant_inlet_temperature, condition.coolant_pressure
        )
        reference_cell_masses = (
            np.asarray(water.density)
            * np.asarray(wall_transfer.flow_area_per_rod)
            * np.asarray(history.zone_lengths)
        )
        balance = LiquidChannelHeatBalance(
            reference_cell_masses,
            channel_id=f"fuel-channel:{history.history_id}:{qualification_profile.profile_id}",
        )
        self.history = history
        self.wall_transfer = wall_transfer
        self.balance = balance
        self.initial_state = jnp.broadcast_to(
            water.specific_enthalpy, (history.zone_count,)
        )
        self.qualification_profile = qualification_profile
        self.participant_id = canonical_fingerprint(
            {
                "kind": "coupled-fuel-thermal-channel",
                "history": history.history_id,
                "wall_transfer": wall_transfer.law_id,
                "balance": balance.channel_id,
                "profile": qualification_profile.profile_id,
            }
        )

    def bulk_temperature(self, channel_state: Array, time: Array, /) -> Array:
        """Bulk temperature [K] of every channel zone at ``time`` [s]."""

        return self._bulk_temperature(channel_state, self.history.conditions(time))

    def _bulk_temperature(
        self, channel_state: Array, condition: IrradiationConditions, /
    ) -> Array:
        return self.wall_transfer.bulk_temperature(
            pressure=condition.coolant_pressure, specific_enthalpy=channel_state
        )

    @staticmethod
    def support_tuple(
        history: IrradiationHistory, wall_transfer: SingleChannelCoolant, /
    ) -> SupportTuple:
        """Channel binding that the trusted release index must qualify."""

        return SupportTuple(
            "thermofluids.channel-heat",
            {
                "history": history.history_id,
                "wall-transfer": wall_transfer.law_id,
                "cells": history.zone_count,
                "water": "iapws-if97-region-1",
                "spatial-method": "reference-mass-upwind",
            },
        )

    def wall_heat_flux(
        self, channel_state: Array, wall_temperature: Array, time: Array, /
    ) -> Array:
        """Heat flux [W/m^2] from the cladding wall into each channel zone."""

        condition = self.history.conditions(time)
        return self.wall_transfer.wall_heat_flux(
            wall_temperature=wall_temperature,
            bulk_temperature=self._bulk_temperature(channel_state, condition),
            pressure=condition.coolant_pressure,
            mass_flux=condition.coolant_mass_flux,
        )

    def residual(
        self,
        channel_state: Array,
        previous_channel_state: Array,
        wall_temperature: Array,
        wall_heat_flux: Array,
        time: Array,
        time_step: Array,
        /,
        *,
        deposited_direct_power: Array,
        interface_areas: Array,
    ) -> Array:
        """Channel energy residual [W], sharing the rod's opposite wall exchange.

        ``time`` is the candidate endpoint: prescribed flow, inlet enthalpy,
        and pressure are evaluated at that same backward-Euler endpoint.
        """

        if wall_temperature.shape != channel_state.shape:
            raise ValueError("Channel temperatures must match the bound interface zones.")
        if interface_areas.shape != channel_state.shape:
            raise ValueError("Current interface areas must match the channel zones.")
        condition = self.history.conditions(time)
        inlet = iapws97_liquid_enthalpy(
            condition.coolant_inlet_temperature, condition.coolant_pressure
        )
        return self.balance.residual(
            channel_state,
            previous_channel_state,
            time_step,
            mass_flow=condition.coolant_mass_flux * self.wall_transfer.flow_area_per_rod,
            inlet_enthalpy=inlet,
            wall_power=wall_heat_flux * interface_areas,
            direct_power=deposited_direct_power,
        )

    def admissible(self, channel_state: Array, time: Array, /) -> Array:
        """Whether every channel zone is inside the single-channel model range."""

        condition = self.history.conditions(time)
        return jnp.all(
            self.wall_transfer.admissible(
                pressure=condition.coolant_pressure,
                mass_flux=condition.coolant_mass_flux,
                specific_enthalpy=channel_state,
            )
        )


__all__ = ["CoupledChannelCoolant"]
