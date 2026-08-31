#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class HodgeSubspaceTracking(StrictModule, NonTrainableState):
    """Metric-aware principal-angle evidence between two common-space subspaces."""

    principal_angles: Array
    projector_residual: Array
    rank_changed: Array
    source_dimension: int = eqx.field(static=True)
    target_dimension: int = eqx.field(static=True)
    tracking_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_basis: ArrayLike,
        target_basis: ArrayLike,
        metric: ArrayLike,
        /,
        *,
        source_id: str,
        target_id: str,
    ):
        source = jnp.asarray(source_basis)
        target = jnp.asarray(target_basis)
        pairing = jnp.asarray(metric)
        if source.ndim != 2 or target.ndim != 2 or pairing.ndim != 2:
            raise ValueError("Hodge tracking bases and metric must be matrices.")
        if source.shape[0] != target.shape[0] or pairing.shape != (
            source.shape[0],
            source.shape[0],
        ):
            raise ValueError("Hodge tracking values do not share one ambient space.")
        source_gram = jnp.conj(source.T) @ pairing @ source
        target_gram = jnp.conj(target.T) @ pairing @ target
        source_scale = _inverse_sqrt(source_gram)
        target_scale = _inverse_sqrt(target_gram)
        source_orthogonal = source @ source_scale
        target_orthogonal = target @ target_scale
        overlap = jnp.conj(source_orthogonal.T) @ pairing @ target_orthogonal
        singular = jnp.linalg.svd(overlap, compute_uv=False)
        singular = jnp.clip(jnp.real(singular), 0.0, 1.0)
        angles = jnp.arccos(singular)
        source_projector = source_orthogonal @ jnp.conj(source_orthogonal.T) @ pairing
        target_projector = target_orthogonal @ jnp.conj(target_orthogonal.T) @ pairing
        self.principal_angles = angles
        self.projector_residual = jnp.linalg.norm(source_projector - target_projector)
        self.rank_changed = jnp.asarray(source.shape[1] != target.shape[1])
        self.source_dimension = int(source.shape[1])
        self.target_dimension = int(target.shape[1])
        self.tracking_id = canonical_fingerprint(
            {
                "kind": "hodge-subspace-tracking",
                "source": str(source_id),
                "target": str(target_id),
                "source_dimension": int(source.shape[1]),
                "target_dimension": int(target.shape[1]),
            }
        )


def _inverse_sqrt(matrix: Array, /) -> Array:
    values, vectors = jnp.linalg.eigh(matrix)
    if not bool(jnp.all(values > 0)):
        raise ValueError("Hodge tracking basis Gram matrix must be positive definite.")
    return (vectors / jnp.sqrt(values)[None, :]) @ jnp.conj(vectors.T)


__all__ = ["HodgeSubspaceTracking"]
