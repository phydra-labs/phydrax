#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._trainable import ParameterOwner
from ._posterior import AbstractBijector


class AffineFlowBijector(AbstractBijector, ParameterOwner):
    """Exact trainable affine chart under the Phydrax bijector contract."""

    shift: Array
    scale: Array
    event_shape: tuple[int, ...] = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        shift: ArrayLike,
        scale: ArrayLike,
        /,
        *,
        architecture_id: str = "native-affine-flow",
    ):
        shift_ = np.asarray(shift)
        scale_ = np.asarray(scale)
        if shift_.shape != scale_.shape or not shift_.shape:
            raise ValueError(
                "Affine flow shift and scale require one matching event shape."
            )
        if not np.all(np.isfinite(shift_)) or np.any(
            ~np.isfinite(scale_) | (scale_ <= 0.0)
        ):
            raise ValueError("Affine flow parameters must be finite with positive scale.")
        identifier = str(architecture_id).strip()
        if not identifier:
            raise ValueError("architecture_id must be non-empty.")
        self.shift = jnp.asarray(shift_)
        self.scale = jnp.asarray(scale_, dtype=self.shift.dtype)
        self.event_shape = tuple(shift_.shape)
        self.architecture_id = identifier
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "native-affine-flow-bijector",
                "architecture": identifier,
                "event_shape": list(self.event_shape),
                "shift": array_tree_fingerprint(shift_),
                "scale": array_tree_fingerprint(scale_),
            }
        )

    def forward_shape(self, raw_shape: tuple[int, ...], /) -> tuple[int, ...]:
        shape = tuple(raw_shape)
        if shape != self.event_shape:
            raise ValueError(
                f"Expected affine flow event shape {self.event_shape}; got {shape}."
            )
        return shape

    def inverse_shape(self, physical_shape: tuple[int, ...], /) -> tuple[int, ...]:
        return self.forward_shape(physical_shape)

    def forward(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value, dtype=self.shift.dtype)
        return self.shift + self.scale * array

    def inverse(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value, dtype=self.shift.dtype)
        return (array - self.shift) / self.scale

    def forward_log_det_jacobian(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        if array.shape != self.event_shape:
            raise ValueError("Affine flow value has an incompatible event shape.")
        return jnp.sum(jnp.log(self.scale))


__all__ = ["AffineFlowBijector"]
