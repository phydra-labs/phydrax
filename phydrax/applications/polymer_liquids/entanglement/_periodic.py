#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.ein import contract

from ...._strict import StrictModule
from ....linalg import inverse
from ._snapshot import PrimitivePathSnapshot


class PeriodicPrimitivePathEvidence(StrictModule):
    chain_winding: Array
    winding_residual: Array
    maximum_winding_residual: Array
    bond_fractional_span: Array
    finite: Array
    successful: Array
    snapshot_id: str = eqx.field(static=True)


def periodic_primitive_path_evidence(
    snapshot: PrimitivePathSnapshot,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> PeriodicPrimitivePathEvidence:
    if not isinstance(snapshot, PrimitivePathSnapshot):
        raise TypeError("snapshot must be PrimitivePathSnapshot.")
    if snapshot.cell_vectors.shape != (3, 3):
        raise ValueError("Periodic primitive-path evidence requires a full 3-D cell.")
    inverse_result = inverse(snapshot.cell_vectors)
    inverse_vectors = inverse_result.value
    chain_positions = snapshot.unwrapped_positions[snapshot.chain_indices]
    segment = chain_positions[:, 1:, :] - chain_positions[:, :-1, :]
    segment_mask = snapshot.chain_mask[:, 1:] & snapshot.chain_mask[:, :-1]
    fractional_segment = contract("cni,ij->cnj", segment, inverse_vectors)
    fractional_sum = jnp.sum(
        jnp.where(segment_mask[..., None], fractional_segment, 0.0), axis=1
    )
    last_index = jnp.sum(snapshot.chain_mask, axis=1).astype(jnp.int32) - 1
    last_position = chain_positions[jnp.arange(chain_positions.shape[0]), last_index]
    endpoint_fractional = contract(
        "ci,ij->cj",
        last_position - chain_positions[:, 0, :],
        inverse_vectors,
    )
    residual = fractional_sum - endpoint_fractional
    winding = jnp.rint(endpoint_fractional).astype(jnp.int32)
    maximum_span = jnp.max(
        jnp.where(
            segment_mask,
            jnp.sqrt(jnp.sum(fractional_segment * fractional_segment, axis=-1)),
            0.0,
        ),
        axis=1,
    )
    maximum_residual = jnp.max(jnp.abs(residual))
    finite = (
        jnp.all(jnp.isfinite(jnp.where(segment_mask[..., None], fractional_segment, 0.0)))
        & jnp.all(jnp.isfinite(endpoint_fractional))
        & jnp.isfinite(maximum_residual)
        & inverse_result.successful
    )
    successful = finite & (maximum_residual <= float(tolerance))
    return PeriodicPrimitivePathEvidence(
        winding,
        residual,
        maximum_residual,
        maximum_span,
        finite,
        successful,
        snapshot.snapshot_id,
    )


__all__ = ["PeriodicPrimitivePathEvidence", "periodic_primitive_path_evidence"]
