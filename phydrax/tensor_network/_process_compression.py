#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule
from ._causal_process import CausalProcessTensor, CombLegSpec


class ProcessMemoryProjectionResult(StrictModule):
    process: CausalProcessTensor
    discarded_initial_weight: Array
    maximum_channel_completeness_residual: Array
    retained_initial_weight: Array
    maximum_invariant_subspace_leakage: Array
    valid: Array

    def __init__(
        self,
        process: CausalProcessTensor,
        discarded_initial_weight: Array,
        maximum_channel_completeness_residual: Array,
        retained_initial_weight: Array,
        maximum_invariant_subspace_leakage: Array,
        /,
    ) -> None:
        self.process = process
        self.discarded_initial_weight = jnp.asarray(discarded_initial_weight)
        self.maximum_channel_completeness_residual = jnp.asarray(
            maximum_channel_completeness_residual
        )
        self.retained_initial_weight = jnp.asarray(retained_initial_weight)
        self.maximum_invariant_subspace_leakage = jnp.asarray(
            maximum_invariant_subspace_leakage
        )
        self.valid = (
            process.valid
            & jnp.isfinite(self.discarded_initial_weight)
            & (self.discarded_initial_weight >= 0.0)
            & jnp.isfinite(self.retained_initial_weight)
            & (self.retained_initial_weight > 0.0)
            & jnp.isfinite(self.maximum_invariant_subspace_leakage)
            & (self.maximum_invariant_subspace_leakage <= 1e-8)
            & (self.maximum_channel_completeness_residual <= 1e-8)
        )


def project_process_memory_subspace(
    process: CausalProcessTensor,
    retained_memory_dimension: int,
    /,
) -> ProcessMemoryProjectionResult:
    """Project onto a fixed memory subspace and validate CPTP after compression."""
    retained = int(retained_memory_dimension)
    old = process.spec.memory_dimension
    if not 1 <= retained <= old:
        raise ValueError("Retained memory dimension is outside the process memory.")
    if retained == old:
        return ProcessMemoryProjectionResult(
            process,
            jnp.asarray(0.0),
            jnp.max(process.channel_completeness_residuals),
            jnp.asarray(1.0),
            jnp.asarray(0.0),
        )
    system = process.spec.system_dimension
    initial = process.initial_state.reshape(system, old, system, old)[
        :, :retained, :, :retained
    ].reshape(system * retained, system * retained)
    # Dense selectors propagated nonfinite discarded entries into the projection.
    initial = jnp.where(jnp.all(jnp.isfinite(process.initial_state)), initial, jnp.nan)
    channel_finite = tuple(
        jnp.all(jnp.isfinite(kraus), axis=(1, 2)) for kraus in process.channel_kraus
    )
    leakage = jnp.stack(
        [
            jnp.where(
                finite,
                jnp.sqrt(
                    jnp.sum(
                        jnp.abs(
                            operator.reshape(system, old, system, old)[
                                :, retained:, :, :retained
                            ]
                        )
                        ** 2
                    )
                ),
                jnp.nan,
            )
            for kraus, finite_kraus in zip(
                process.channel_kraus, channel_finite, strict=True
            )
            for operator, finite in zip(kraus, finite_kraus, strict=True)
        ]
    )
    maximum_leakage = jnp.max(leakage)
    retained_weight = jnp.real(jnp.trace(initial))
    initial = initial / jnp.where(retained_weight > 0.0, retained_weight, 1.0)
    channels = tuple(
        jnp.where(
            finite[:, None, None],
            kraus.reshape(kraus.shape[0], system, old, system, old)[
                :, :, :retained, :, :retained
            ].reshape(kraus.shape[0], system * retained, system * retained),
            jnp.nan,
        )
        for kraus, finite in zip(process.channel_kraus, channel_finite, strict=True)
    )
    compressed = CausalProcessTensor(
        CombLegSpec(
            system,
            retained,
            process.spec.slot_count,
            vectorization=process.spec.vectorization,
            spec_id=f"{process.spec.spec_id}:memory-{retained}",
        ),
        initial,
        channels,
        process_id=f"{process.process_id}:memory-{retained}",
    )
    return ProcessMemoryProjectionResult(
        compressed,
        1.0 - retained_weight,
        jnp.max(compressed.channel_completeness_residuals),
        retained_weight,
        maximum_leakage,
    )


__all__ = ["ProcessMemoryProjectionResult", "project_process_memory_subspace"]
