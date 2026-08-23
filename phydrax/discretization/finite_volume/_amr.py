#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..amr import (
    ConservativeAMRSubcyclingPlan,
    ConservativeBlockTransfer,
    FluxRegister,
)


class ConservativeAMRSynchronizationResult(StrictModule):
    coarse_state: Array
    fine_state: Array
    restricted_fine_state: Array
    flux_register: FluxRegister
    covered_mask: Array
    conservation_defect: Array


class ConservativeAMRSynchronizationPlan(StrictModule, NonTrainableState):
    """Subcycle, reflux, and restrict one fixed two-level hierarchy."""

    subcycling: ConservativeAMRSubcyclingPlan
    transfer: ConservativeBlockTransfer
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_dimensions: int,
        refinement_ratio: int = 2,
        /,
        *,
        temporal_method_id: str = "temporal:caller-supplied",
    ):
        self.subcycling = ConservativeAMRSubcyclingPlan(
            refinement_ratio,
            temporal_method_id=temporal_method_id,
        )
        self.transfer = ConservativeBlockTransfer(
            spatial_dimensions, refinement_ratio
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-amr-synchronization",
                "subcycling": self.subcycling.plan_id,
                "transfer": self.transfer.transfer_id,
            }
        )

    def advance(
        self,
        time: ArrayLike,
        coarse_state: ArrayLike,
        fine_state: ArrayLike,
        step_size: ArrayLike,
        coarse_step: Callable[[Array, Array, Array, Any], Array],
        fine_step: Callable[[Array, Array, Array, Any], Array],
        coarse_flux: Callable[[Array, Any], Array],
        fine_flux: Callable[[Array, Any], Array],
        restrict_flux: Callable[[Array], Array],
        interface_mask: ArrayLike,
        covered_mask: ArrayLike,
        coarse_volume: ArrayLike,
        args: Any = None,
        /,
    ) -> ConservativeAMRSynchronizationResult:
        subcycled = self.subcycling.advance(
            time,
            coarse_state,
            fine_state,
            step_size,
            coarse_step,
            fine_step,
            coarse_flux,
            fine_flux,
            restrict_flux,
            interface_mask,
            coarse_volume,
            args,
        )
        restricted = self.transfer.restrict(subcycled.fine_state)
        mask = jnp.asarray(covered_mask, dtype=bool)
        if mask.shape != subcycled.coarse_state.shape[: mask.ndim]:
            raise ValueError("Covered-cell mask must align with coarse state.")
        broadcast = mask.reshape(
            mask.shape + (1,) * (subcycled.coarse_state.ndim - mask.ndim)
        )
        synchronized = jnp.where(broadcast, restricted, subcycled.coarse_state)
        defect = jnp.sum(
            jnp.where(broadcast, synchronized - restricted, 0.0),
            axis=tuple(range(mask.ndim)),
        )
        return ConservativeAMRSynchronizationResult(
            coarse_state=synchronized,
            fine_state=subcycled.fine_state,
            restricted_fine_state=restricted,
            flux_register=subcycled.flux_register,
            covered_mask=mask,
            conservation_defect=defect,
        )


def flux_register_from_accepted_steps(
    coarse_result: Any,
    fine_results: tuple[Any, ...],
    axis: int,
    restrict_flux: Callable[[Array], Array],
    interface_mask: ArrayLike,
    /,
) -> FluxRegister:
    """Build a register from accepted SSPRK time-averaged face fluxes."""
    axis_ = int(axis)
    coarse_flux = (
        coarse_result.accepted_step_size
        * coarse_result.accepted_integrated_fluxes[axis_]
    )
    fine_flux = sum(
        (
            result.accepted_step_size
            * result.accepted_integrated_fluxes[axis_]
            for result in fine_results
        ),
        jnp.zeros_like(fine_results[0].accepted_integrated_fluxes[axis_]),
    )
    restricted_fine = restrict_flux(fine_flux)
    if restricted_fine.shape != coarse_flux.shape:
        raise ValueError("Restricted accepted fine flux must match coarse flux.")
    return FluxRegister(
        coarse_flux,
        restricted_fine,
        interface_mask,
        accumulated_time=coarse_result.accepted_step_size,
        refinement_ratio=len(fine_results),
        register_id=canonical_fingerprint(
            {
                "kind": "accepted-fv-flux-register",
                "axis": axis_,
                "fine_steps": len(fine_results),
            }
        ),
    )




__all__ = [
    "ConservativeAMRSynchronizationPlan",
    "ConservativeAMRSynchronizationResult",
    "flux_register_from_accepted_steps",
]
