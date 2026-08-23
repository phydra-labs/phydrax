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
    """Time-integrated coarse/fine interface flux mismatch."""

    coarse_flux: Array
    fine_flux: Array
    interface_mask: Array
    accumulated_time: Array
    orientation: int = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    register_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse_flux: ArrayLike,
        fine_flux: ArrayLike,
        interface_mask: ArrayLike,
        *,
        accumulated_time: ArrayLike = 1.0,
        orientation: int = 1,
        refinement_ratio: int = 1,
        register_id: str | None = None,
    ):
        coarse = jnp.asarray(coarse_flux)
        fine = jnp.asarray(fine_flux)
        mask = jnp.asarray(interface_mask, dtype=bool)
        time = jnp.asarray(accumulated_time).reshape(())
        orientation_ = int(orientation)
        ratio = int(refinement_ratio)
        if coarse.shape != fine.shape or mask.shape != coarse.shape[: mask.ndim]:
            raise ValueError("Flux register coarse/fine/mask shapes must align.")
        if orientation_ not in (-1, 1) or ratio <= 0:
            raise ValueError("Flux-register orientation and refinement ratio are invalid.")
        coarse = eqx.error_if(
            coarse,
            jnp.any(~jnp.isfinite(coarse)) | jnp.any(~jnp.isfinite(fine)),
            "Flux-register values must be finite.",
        )
        time = eqx.error_if(
            time,
            ~jnp.isfinite(time) | (time <= 0.0),
            "Flux-register accumulation time must be finite and positive.",
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "integrated-flux-register",
                    "shape": list(coarse.shape),
                    "mask_shape": list(mask.shape),
                    "orientation": orientation_,
                    "refinement_ratio": ratio,
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
        self.accumulated_time = time
        self.orientation = orientation_
        self.refinement_ratio = ratio
        self.register_id = identifier

    def mismatch(self, /) -> Array:
        mask = self.interface_mask.reshape(
            self.interface_mask.shape
            + (1,) * (self.coarse_flux.ndim - self.interface_mask.ndim)
        )
        difference = self.orientation * (self.fine_flux - self.coarse_flux)
        return jnp.where(mask, difference, 0.0)

    def correction(self, cell_volume: ArrayLike, /) -> Array:
        volume = jnp.asarray(cell_volume)
        mismatch = self.mismatch()
        if volume.shape != mismatch.shape[: volume.ndim]:
            raise ValueError("Cell volume must align with reflux correction.")
        volume = eqx.error_if(
            volume,
            jnp.any(~jnp.isfinite(volume) | (volume <= 0.0)),
            "Reflux cell volumes must be finite and positive.",
        )
        reshape = volume.shape + (1,) * (mismatch.ndim - volume.ndim)
        return mismatch / volume.reshape(reshape)

    def apply(self, coarse_state: ArrayLike, cell_volume: ArrayLike, /) -> Array:
        state = jnp.asarray(coarse_state)
        correction = self.correction(cell_volume)
        if state.shape != correction.shape:
            raise ValueError("Coarse state and reflux correction must align.")
        return state + correction


__all__ = ["FluxRegister"]
