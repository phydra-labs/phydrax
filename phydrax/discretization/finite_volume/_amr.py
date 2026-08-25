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
from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..amr import (
    ConservativeAMRSubcyclingPlan,
    ConservativeBlockTransfer,
    FluxRegister,
)
from ._flux_ledger import FiniteVolumeAcceptedFluxIntegralLedger
from ._precision import FiniteVolumePrecisionPolicy


def _accepted_result_flag(result: Any, name: str, /) -> Array:
    if not hasattr(result, "accepted"):
        raise TypeError(f"{name} must expose a scalar boolean accepted flag.")
    accepted = jnp.asarray(result.accepted)
    if accepted.shape != () or accepted.dtype != jnp.dtype(bool):
        raise TypeError(f"{name} must expose a scalar boolean accepted flag.")
    return accepted


def _time_tolerance(first: Array, second: Array, /) -> Array:
    dtype = jnp.result_type(first, second, jnp.asarray(1.0))
    first_ = first.astype(dtype)
    second_ = second.astype(dtype)
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=dtype),
        jnp.maximum(jnp.abs(first_), jnp.abs(second_)),
    )
    return jnp.asarray(16.0, dtype=dtype) * jnp.finfo(dtype).eps * scale


class ConservativeAMRSynchronizationResult(StrictModule):
    coarse_state: Array
    fine_state: Array
    restricted_fine_state: Array
    flux_register: FluxRegister
    covered_mask: Array
    conservation_defect: Array
    precision_evidence: PrecisionEvidenceEnvelope


class ConservativeAMRSynchronizationPlan(StrictModule, NonTrainableState):
    """Subcycle, reflux, and restrict one fixed two-level hierarchy."""

    subcycling: ConservativeAMRSubcyclingPlan
    transfer: ConservativeBlockTransfer
    precision: FiniteVolumePrecisionPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        spatial_dimensions: int,
        refinement_ratio: int = 2,
        /,
        *,
        temporal_method_id: str = "temporal:caller-supplied",
        precision: FiniteVolumePrecisionPolicy | None = None,
    ):
        precision_ = FiniteVolumePrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, FiniteVolumePrecisionPolicy):
            raise TypeError("precision must be a FiniteVolumePrecisionPolicy.")
        self.subcycling = ConservativeAMRSubcyclingPlan(
            refinement_ratio,
            temporal_method_id=temporal_method_id,
            precision=precision_,
        )
        self.transfer = ConservativeBlockTransfer(spatial_dimensions, refinement_ratio)
        self.precision = precision_
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
        restricted = self.precision.storage(self.transfer.restrict(subcycled.fine_state))
        mask = jnp.asarray(covered_mask, dtype=bool)
        if mask.shape != subcycled.coarse_state.shape[: mask.ndim]:
            raise ValueError("Covered-cell mask must align with coarse state.")
        broadcast = mask.reshape(
            mask.shape + (1,) * (subcycled.coarse_state.ndim - mask.ndim)
        )
        synchronized = self.precision.storage(
            jnp.where(broadcast, restricted, subcycled.coarse_state)
        )
        defect = jnp.sum(
            self.precision.reduction(
                jnp.where(broadcast, synchronized - restricted, 0.0)
            ),
            axis=tuple(range(mask.ndim)),
        )
        return ConservativeAMRSynchronizationResult(
            coarse_state=synchronized,
            fine_state=subcycled.fine_state,
            restricted_fine_state=restricted,
            flux_register=subcycled.flux_register,
            covered_mask=mask,
            conservation_defect=defect,
            precision_evidence=self.precision.evidence(),
        )


def flux_register_from_accepted_steps(
    coarse_result: Any,
    fine_results: tuple[Any, ...],
    axis: int,
    restrict_flux: Callable[[Array], Array],
    interface_mask: ArrayLike,
    /,
    *,
    precision: FiniteVolumePrecisionPolicy | None = None,
) -> FluxRegister:
    """Build a register directly from accepted SSPRK content integrals."""
    precision_ = FiniteVolumePrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, FiniteVolumePrecisionPolicy):
        raise TypeError("precision must be a FiniteVolumePrecisionPolicy.")
    if not fine_results:
        raise ValueError("fine_results must contain at least one accepted step.")
    coarse_accepted = _accepted_result_flag(coarse_result, "coarse_result")
    fine_accepted = tuple(
        _accepted_result_flag(result, f"fine_results[{index}]")
        for index, result in enumerate(fine_results)
    )
    coarse_ledger = coarse_result.accepted_flux_integrals
    fine_ledgers = tuple(result.accepted_flux_integrals for result in fine_results)
    if not isinstance(coarse_ledger, FiniteVolumeAcceptedFluxIntegralLedger) or any(
        not isinstance(ledger, FiniteVolumeAcceptedFluxIntegralLedger)
        for ledger in fine_ledgers
    ):
        raise TypeError("AMR reflux requires accepted flux-integral ledgers.")
    axis_ = int(axis)
    if axis_ < 0 or axis_ >= len(coarse_ledger.blocks):
        raise ValueError("axis does not select a coarse accepted flux block.")
    if any(len(ledger.blocks) != len(coarse_ledger.blocks) for ledger in fine_ledgers):
        raise ValueError("Coarse and fine accepted ledgers must align by flux block.")
    expected_fine_ledger_id = fine_ledgers[0].ledger_id
    if any(ledger.ledger_id != expected_fine_ledger_id for ledger in fine_ledgers[1:]):
        raise ValueError(
            "Fine accepted ledgers must have identical block IDs and routes."
        )
    coarse_block = coarse_ledger.blocks[axis_]
    fine_blocks = tuple(ledger.blocks[axis_] for ledger in fine_ledgers)
    if (
        coarse_ledger.units != "content"
        or any(ledger.units != coarse_ledger.units for ledger in fine_ledgers)
        or any(
            block.block_kind != coarse_block.block_kind
            or block.component_shape != coarse_block.component_shape
            for block in fine_blocks
        )
    ):
        raise ValueError("Coarse and fine accepted flux blocks are incompatible.")
    coarse_flux = precision_.reduction(coarse_block.flux_integral)
    coarse_flux = eqx.error_if(
        coarse_flux,
        ~coarse_accepted,
        "AMR reflux requires a successful accepted coarse result.",
    )
    for index, accepted in enumerate(fine_accepted):
        coarse_flux = eqx.error_if(
            coarse_flux,
            ~accepted,
            f"AMR reflux requires successful accepted fine results; result {index} failed.",
        )
    coarse_flux = eqx.error_if(
        coarse_flux,
        jnp.abs(fine_ledgers[0].start_time - coarse_ledger.start_time)
        > _time_tolerance(fine_ledgers[0].start_time, coarse_ledger.start_time),
        "Fine accepted intervals must start at the coarse interval start.",
    )
    coarse_flux = eqx.error_if(
        coarse_flux,
        jnp.abs(fine_ledgers[-1].end_time - coarse_ledger.end_time)
        > _time_tolerance(fine_ledgers[-1].end_time, coarse_ledger.end_time),
        "Fine accepted intervals must end at the coarse interval end.",
    )
    for index, (previous, current) in enumerate(
        zip(fine_ledgers, fine_ledgers[1:]),
        start=1,
    ):
        boundary_difference = current.start_time - previous.end_time
        tolerance = _time_tolerance(current.start_time, previous.end_time)
        coarse_flux = eqx.error_if(
            coarse_flux,
            boundary_difference > tolerance,
            f"Fine accepted intervals contain a gap before result {index}.",
        )
        coarse_flux = eqx.error_if(
            coarse_flux,
            boundary_difference < -tolerance,
            f"Fine accepted intervals overlap before result {index}.",
        )
        coarse_flux = eqx.error_if(
            coarse_flux,
            current.accepted_step <= previous.accepted_step,
            "Fine accepted-step IDs must be strictly monotone.",
        )
    fine_flux = sum(
        (precision_.reduction(block.flux_integral) for block in fine_blocks),
        jnp.zeros_like(precision_.reduction(fine_blocks[0].flux_integral)),
    )
    restricted_fine = precision_.reduction(restrict_flux(fine_flux))
    if restricted_fine.shape != coarse_flux.shape:
        raise ValueError("Restricted accepted fine flux must match coarse flux.")
    return FluxRegister(
        coarse_flux,
        restricted_fine,
        interface_mask,
        accumulated_time=coarse_ledger.end_time - coarse_ledger.start_time,
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
