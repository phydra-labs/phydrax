#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

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
    ):
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
    selector_memory = jnp.eye(old, dtype=process.initial_state.dtype)[:retained]
    selector = jnp.kron(
        jnp.eye(system, dtype=process.initial_state.dtype), selector_memory
    )
    projector = jnp.conj(selector.T) @ selector
    identity = jnp.eye(system * old, dtype=process.initial_state.dtype)
    leakage = jnp.stack(
        [
            jnp.sqrt(jnp.sum(jnp.abs((identity - projector) @ operator @ projector) ** 2))
            for kraus in process.channel_kraus
            for operator in kraus
        ]
    )
    maximum_leakage = jnp.max(leakage)
    initial = selector @ process.initial_state @ jnp.conj(selector.T)
    retained_weight = jnp.real(jnp.trace(initial))
    initial = initial / jnp.where(retained_weight > 0.0, retained_weight, 1.0)
    channels = tuple(
        jnp.stack([selector @ operator @ jnp.conj(selector.T) for operator in kraus])
        for kraus in process.channel_kraus
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
