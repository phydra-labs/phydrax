#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..materials import ConservativeMaterialTransfer
from ._activation import MaterialActivationState
from ._runtime import ManufacturingRuntimeState


@dataclass(frozen=True, slots=True)
class ProcessTransferPlan:
    """Conservative transfer of intensive and extensive process state."""

    material_transfer: ConservativeMaterialTransfer

    @classmethod
    def create(
        cls,
        matrix: ArrayLike,
        source_control_volumes: ArrayLike,
        target_control_volumes: ArrayLike,
        /,
        *,
        tolerance: float = 1e-10,
    ) -> ProcessTransferPlan:
        return cls(
            ConservativeMaterialTransfer.create(
                matrix,
                source_control_volumes,
                target_control_volumes,
                tolerance=tolerance,
            )
        )

    def transfer_intensive(self, field: ArrayLike, /) -> Array:
        return self.material_transfer.apply(field)

    def transfer_extensive(self, field: ArrayLike, /) -> Array:
        values = jnp.asarray(field)
        source_weights = self.material_transfer.source_weights
        target_weights = self.material_transfer.target_weights
        if values.ndim < 1 or values.shape[0] != source_weights.size:
            raise ValueError("Extensive process field does not match source topology.")
        density = values / source_weights[(...,) + (None,) * (values.ndim - 1)]
        transferred = self.material_transfer.apply(density)
        return transferred * target_weights[(...,) + (None,) * (transferred.ndim - 1)]

    def transfer_runtime_state(
        self, state: ManufacturingRuntimeState, /, *, activation_tolerance: float = 0.0
    ) -> ManufacturingRuntimeState:
        mass = self.transfer_extensive(state.deposited_mass_kg)
        energy = self.transfer_extensive(state.supplied_energy_j)
        active_fraction = self.transfer_intensive(state.activation.active.astype(float))
        activation_time = self.transfer_intensive(
            jnp.where(
                state.activation.active,
                state.activation.activation_time_s,
                state.time_s,
            )
        )
        active = active_fraction > float(activation_tolerance)
        return ManufacturingRuntimeState(
            state.time_s,
            mass,
            energy,
            MaterialActivationState(
                active,
                jnp.where(active, activation_time, -jnp.inf),
            ),
        )


__all__ = ["ProcessTransferPlan"]
