#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from flowjax.bijections import AbstractBijection
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ._posterior import AbstractBijector


class FlowJAXBijectionAdapter(AbstractBijector):
    """Unconditional exact FlowJAX bijection under the PhydraX bijector contract."""

    bijection: AbstractBijection
    event_shape: tuple[int, ...] = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)
    adapter_id: str = eqx.field(static=True)

    def __init__(
        self,
        bijection: AbstractBijection,
        /,
        *,
        architecture_id: str,
    ):
        if not isinstance(bijection, AbstractBijection):
            raise TypeError("bijection must be a FlowJAX AbstractBijection.")
        if bijection.cond_shape is not None:
            raise ValueError("Targeted FlowJAX bijections must be unconditional.")
        shape = tuple(int(size) for size in bijection.shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("FlowJAX bijection requires a non-empty event shape.")
        identifier = str(architecture_id).strip()
        if not identifier:
            raise ValueError("architecture_id must be non-empty.")
        self.bijection = bijection
        self.event_shape = shape
        self.architecture_id = identifier
        self.adapter_id = canonical_fingerprint(
            {
                "kind": "flowjax-bijection-adapter",
                "architecture": identifier,
                "event_shape": list(shape),
                "condition": None,
            }
        )

    def forward_shape(self, raw_shape: tuple[int, ...], /) -> tuple[int, ...]:
        shape = tuple(int(size) for size in raw_shape)
        if shape != self.event_shape:
            raise ValueError(
                f"Expected FlowJAX event shape {self.event_shape}; got {shape}."
            )
        return shape

    def inverse_shape(self, physical_shape: tuple[int, ...], /) -> tuple[int, ...]:
        return self.forward_shape(physical_shape)

    def forward(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        return self.bijection.transform_and_log_det(array)[0]

    def inverse(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        return self.bijection.inverse_and_log_det(array)[0]

    def forward_log_det_jacobian(self, value: ArrayLike, /) -> Array:
        array = jnp.asarray(value)
        return self.bijection.transform_and_log_det(array)[1]


__all__ = ["FlowJAXBijectionAdapter"]
