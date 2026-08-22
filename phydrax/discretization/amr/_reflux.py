#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class FluxRegister(StrictModule):
    """Coarse/fine interface flux mismatch accumulated over one synchronization step."""

    coarse_flux: Array
    fine_flux: Array
    interface_mask: Array
    register_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse_flux: ArrayLike,
        fine_flux: ArrayLike,
        interface_mask: ArrayLike,
        *,
        register_id: str | None = None,
    ):
        coarse = jnp.asarray(coarse_flux)
        fine = jnp.asarray(fine_flux)
        mask = jnp.asarray(interface_mask, dtype=bool)
        if coarse.shape != fine.shape or mask.shape != coarse.shape[: mask.ndim]:
            raise ValueError("Flux register coarse/fine/mask shapes must align.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "flux-register",
                    "shape": list(coarse.shape),
                    "mask_shape": list(mask.shape),
                }
            )
            if register_id is None
            else str(register_id)
        )
        if not identifier:
            raise ValueError("register_id must be non-empty.")
        self.coarse_flux = coarse
        self.fine_flux = fine
        self.interface_mask = mask
        self.register_id = identifier

    def mismatch(self, /) -> Array:
        mask = self.interface_mask.reshape(
            self.interface_mask.shape
            + (1,) * (self.coarse_flux.ndim - self.interface_mask.ndim)
        )
        return jnp.where(mask, self.fine_flux - self.coarse_flux, 0.0)

    def correction(self, cell_volume: ArrayLike, /) -> Array:
        volume = jnp.asarray(cell_volume)
        mismatch = self.mismatch()
        if volume.shape != mismatch.shape[: volume.ndim]:
            raise ValueError("Cell volume must align with reflux correction.")
        reshape = volume.shape + (1,) * (mismatch.ndim - volume.ndim)
        return mismatch / volume.reshape(reshape)

    def apply(self, coarse_state: ArrayLike, cell_volume: ArrayLike, /) -> Array:
        state = jnp.asarray(coarse_state)
        correction = self.correction(cell_volume)
        if state.shape != correction.shape:
            raise ValueError("Coarse state and reflux correction must align.")
        return state + correction


__all__ = ["FluxRegister"]
