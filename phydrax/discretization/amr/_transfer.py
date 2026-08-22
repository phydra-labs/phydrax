#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ConservativeBlockTransfer(StrictModule, NonTrainableState):
    """Constant-preserving prolongation and conservative cell-average restriction."""

    spatial_dimensions: int = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(self, spatial_dimensions: int, refinement_ratio: int = 2, /):
        dimensions = int(spatial_dimensions)
        ratio = int(refinement_ratio)
        if dimensions <= 0 or ratio <= 1:
            raise ValueError("Transfer dimensions and refinement ratio must be valid.")
        self.spatial_dimensions = dimensions
        self.refinement_ratio = ratio
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "conservative-block-transfer",
                "spatial_dimensions": dimensions,
                "refinement_ratio": ratio,
            }
        )

    def prolong(self, coarse: ArrayLike, /) -> Array:
        value = jnp.asarray(coarse)
        if value.ndim < self.spatial_dimensions:
            raise ValueError("Coarse values lack declared spatial dimensions.")
        result = value
        for axis in range(self.spatial_dimensions):
            result = jnp.repeat(result, self.refinement_ratio, axis=axis)
        return result

    def restrict(self, fine: ArrayLike, /) -> Array:
        value = jnp.asarray(fine)
        if value.ndim < self.spatial_dimensions:
            raise ValueError("Fine values lack declared spatial dimensions.")
        shape = list(value.shape)
        for axis in range(self.spatial_dimensions):
            if shape[axis] % self.refinement_ratio:
                raise ValueError("Fine spatial shape must divide by refinement ratio.")
        result = value
        for axis in reversed(range(self.spatial_dimensions)):
            shape = result.shape
            reshaped = (
                shape[:axis]
                + (shape[axis] // self.refinement_ratio, self.refinement_ratio)
                + shape[axis + 1 :]
            )
            result = result.reshape(reshaped).mean(axis=axis + 1)
        return result

    def conservation_residual(self, coarse: ArrayLike, /) -> Array:
        value = jnp.asarray(coarse)
        fine = self.prolong(value)
        restricted = self.restrict(fine)
        return restricted - value


__all__ = ["ConservativeBlockTransfer"]
