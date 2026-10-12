#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative fixed-capacity enthalpy balance of a through-flow liquid channel.

The numerical channel cells own fixed reference fluid masses. Their enthalpy
states evolve by accumulation, directed upstream transport, and declared
integrated heat sources. No time stepping or nonlinear iteration belongs here;
a coupled problem supplies the temporal derivative and solves this residual.
This is a prescribed-mass-flow thermal channel, not compressible hydraulics.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation, SparseLinearMap


@final
class LiquidChannelHeatBalance(StrictModule, NonTrainableState):
    """Upwind enthalpy transport with explicit accumulation and source units.

    Cells are ordered from inlet to outlet. ``reference_cell_masses`` [kg]
    are fixed numerical control-volume masses, not an inferred evolving
    density. Inflow has a prescribed positive ``mass_flow`` [kg/s].
    """

    reference_cell_masses: Array
    upstream: SparseLinearMap
    inlet_route: Array
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_cell_masses: ArrayLike,
        /,
        *,
        channel_id: str,
    ) -> None:
        masses = np.asarray(reference_cell_masses, dtype=np.float64)
        if masses.ndim != 1 or masses.size < 1:
            raise ValueError("A channel requires one reference fluid mass per cell.")
        if np.any(~np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("Channel reference masses must be finite and positive.")
        if not isinstance(channel_id, str) or not channel_id:
            raise ValueError("channel_id must be a non-empty scientific identifier.")
        count = masses.size
        relation = EdgeRelation(
            np.arange(count - 1, dtype=np.int32),
            np.arange(1, count, dtype=np.int32),
            source_size=count,
            target_size=count,
        )
        upstream = SparseLinearMap(relation, np.ones((count - 1,), dtype=np.float64))
        inlet = np.zeros((count,), dtype=np.float64)
        inlet[0] = 1.0
        self.reference_cell_masses = jnp.asarray(masses)
        self.upstream = upstream
        self.inlet_route = jnp.asarray(inlet)
        self.channel_id = canonical_fingerprint(
            {
                "kind": "liquid-channel-enthalpy-balance",
                "scientific_id": channel_id,
                "reference_cell_masses": masses,
            }
        )

    @property
    def cell_count(self) -> int:
        return self.reference_cell_masses.shape[0]

    def residual(
        self,
        enthalpy: Array,
        previous_enthalpy: Array,
        time_step: Array,
        /,
        *,
        mass_flow: Array,
        inlet_enthalpy: Array,
        wall_power: Array,
        direct_power: Array,
    ) -> Array:
        """Cell energy residual [W], positive source power entering the liquid."""

        expected = (self.cell_count,)
        if any(
            value.shape != expected
            for value in (enthalpy, previous_enthalpy, wall_power, direct_power)
        ):
            raise ValueError(
                "Channel state and integrated heat sources must match cells."
            )
        if mass_flow.shape != () or inlet_enthalpy.shape != () or time_step.shape != ():
            raise ValueError("Mass flow, inlet enthalpy and time step must be scalars.")
        time_step = eqx.error_if(
            time_step,
            ~jnp.isfinite(time_step) | (time_step <= 0.0),
            "Channel time step must be finite and positive.",
        )
        mass_flow = eqx.error_if(
            mass_flow,
            ~jnp.isfinite(mass_flow) | (mass_flow <= 0.0),
            "The through-flow thermal channel requires positive finite mass flow.",
        )
        incoming = self.upstream.mv(enthalpy) + self.inlet_route * inlet_enthalpy
        accumulation = (
            self.reference_cell_masses * (enthalpy - previous_enthalpy) / time_step
        )
        return (
            accumulation + mass_flow * (enthalpy - incoming) - wall_power - direct_power
        )

    def stored_energy_change(self, enthalpy: Array, previous_enthalpy: Array, /) -> Array:
        """Total channel thermal energy change [J]."""

        return jnp.sum(self.reference_cell_masses * (enthalpy - previous_enthalpy))

    def outlet_power(
        self, enthalpy: Array, mass_flow: Array, inlet_enthalpy: Array, /
    ) -> Array:
        """Net thermal power carried from inlet to outlet [W]."""

        return mass_flow * (enthalpy[-1] - inlet_enthalpy)


__all__ = ["LiquidChannelHeatBalance"]
