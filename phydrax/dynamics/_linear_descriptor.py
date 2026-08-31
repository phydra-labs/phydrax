#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule


class DescriptorSystemEvidence(StrictModule):
    finite: Array
    mass_rank: Array
    state_rank: Array
    regular_at_probe: Array
    probe: Array


class LinearDescriptorSystem(StrictModule):
    """Linear descriptor dynamics E xdot = A x + B u, y = C x + D u."""

    mass_matrix: Array
    state_matrix: Array
    input_matrix: Array
    output_matrix: Array
    feedthrough_matrix: Array
    evidence: DescriptorSystemEvidence
    system_id: str = eqx.field(static=True)

    def __init__(
        self,
        mass_matrix: ArrayLike,
        state_matrix: ArrayLike,
        input_matrix: ArrayLike,
        output_matrix: ArrayLike,
        feedthrough_matrix: ArrayLike,
        /,
        *,
        regularity_probe: ArrayLike = 1.0,
        system_id: str | None = None,
    ):
        mass, state, inputs, outputs, feedthrough = (
            jnp.asarray(value)
            for value in (
                mass_matrix,
                state_matrix,
                input_matrix,
                output_matrix,
                feedthrough_matrix,
            )
        )
        if mass.ndim < 2 or mass.shape[-2] != mass.shape[-1]:
            raise ValueError("mass_matrix must end in one square matrix.")
        size = int(mass.shape[-1])
        batch = mass.shape[:-2]
        input_count = int(inputs.shape[-1]) if inputs.ndim >= 2 else -1
        output_count = int(outputs.shape[-2]) if outputs.ndim >= 2 else -1
        if (
            state.shape != batch + (size, size)
            or inputs.shape != batch + (size, input_count)
            or outputs.shape != batch + (output_count, size)
            or feedthrough.shape != batch + (output_count, input_count)
        ):
            raise ValueError("Descriptor matrices have incompatible batch dimensions.")
        dtype = jnp.result_type(mass, state, inputs, outputs, feedthrough, jnp.float64)
        mass, state, inputs, outputs, feedthrough = (
            value.astype(dtype) for value in (mass, state, inputs, outputs, feedthrough)
        )
        probe = jnp.asarray(regularity_probe, dtype=dtype)
        if probe.shape != () or bool(jnp.any(~jnp.isfinite(probe))):
            raise ValueError("regularity_probe must be one finite scalar.")
        pencil = probe * mass - state
        finite = all(
            bool(jnp.all(jnp.isfinite(value)))
            for value in (mass, state, inputs, outputs, feedthrough)
        )
        mass_rank = jnp.linalg.matrix_rank(mass)
        state_rank = jnp.linalg.matrix_rank(state)
        regular = jnp.linalg.matrix_rank(pencil) == size
        evidence = DescriptorSystemEvidence(
            jnp.asarray(finite), mass_rank, state_rank, regular, probe
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "linear-descriptor-system",
                    "mass": array_tree_fingerprint(mass),
                    "state": array_tree_fingerprint(state),
                    "inputs": array_tree_fingerprint(inputs),
                    "outputs": array_tree_fingerprint(outputs),
                    "feedthrough": array_tree_fingerprint(feedthrough),
                }
            )
            if system_id is None
            else str(system_id)
        )
        if not identifier:
            raise ValueError("system_id must be non-empty.")
        self.mass_matrix = mass
        self.state_matrix = state
        self.input_matrix = inputs
        self.output_matrix = outputs
        self.feedthrough_matrix = feedthrough
        self.evidence = evidence
        self.system_id = identifier

    @property
    def state_size(self) -> int:
        return int(self.state_matrix.shape[-1])

    @property
    def input_size(self) -> int:
        return int(self.input_matrix.shape[-1])

    @property
    def output_size(self) -> int:
        return int(self.output_matrix.shape[-2])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.state_matrix.shape[:-2])


__all__ = ["DescriptorSystemEvidence", "LinearDescriptorSystem"]
