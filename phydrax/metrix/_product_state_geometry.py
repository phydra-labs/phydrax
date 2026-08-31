#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._state_geometry import AbstractStateGeometry


class ProductStateGeometryBlock(StrictModule, NonTrainableState):
    """One shaped component of a flat product-state representation."""

    geometry: AbstractStateGeometry
    shape: tuple[int, ...] = eqx.field(static=True)
    size: int = eqx.field(static=True)
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: AbstractStateGeometry,
        shape: Sequence[int],
        /,
        *,
        block_id: str,
    ):
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("geometry must be an AbstractStateGeometry.")
        shape_ = tuple(int(value) for value in shape)
        if not shape_ or any(value <= 0 for value in shape_):
            raise ValueError("Product-state block shapes must be nonempty and positive.")
        identifier = str(block_id)
        if not identifier:
            raise ValueError("block_id must be nonempty.")
        self.geometry = geometry
        self.shape = shape_
        self.size = prod(shape_)
        self.block_id = identifier


class ProductStateGeometry(AbstractStateGeometry):
    """Flat array representation of a static product of shaped state geometries."""

    blocks: tuple[ProductStateGeometryBlock, ...]
    offsets: tuple[int, ...] = eqx.field(static=True)
    total_size: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    trivial: bool = eqx.field(static=True)
    supports_exact_pullback: bool = eqx.field(static=True)
    supports_commutator_free: bool = eqx.field(static=True)

    def __init__(
        self,
        blocks: Sequence[ProductStateGeometryBlock],
        /,
        *,
        geometry_id: str | None = None,
    ):
        blocks_ = tuple(blocks)
        if not blocks_ or any(
            not isinstance(block, ProductStateGeometryBlock) for block in blocks_
        ):
            raise TypeError("blocks must contain ProductStateGeometryBlock values.")
        if len({block.block_id for block in blocks_}) != len(blocks_):
            raise ValueError("Product-state block IDs must be unique.")
        offsets = [0]
        for block in blocks_:
            offsets.append(offsets[-1] + block.size)
        identifier = geometry_id or canonical_fingerprint(
            {
                "kind": "product-state-geometry",
                "blocks": [
                    {
                        "id": block.block_id,
                        "shape": list(block.shape),
                        "geometry": block.geometry.geometry_id,
                    }
                    for block in blocks_
                ],
            }
        )
        self.blocks = blocks_
        self.offsets = tuple(offsets)
        self.total_size = offsets[-1]
        self.geometry_id = str(identifier)
        self.retraction_method = "product"
        self.trivial = all(block.geometry.trivial for block in blocks_)
        self.supports_exact_pullback = all(
            block.geometry.supports_exact_pullback for block in blocks_
        )
        self.supports_commutator_free = all(
            block.geometry.supports_commutator_free for block in blocks_
        )

    def _flat(self, value: ArrayLike, name: str, /) -> Array:
        array = jnp.asarray(value)
        if array.shape != (self.total_size,):
            raise ValueError(
                f"{name} must have shape ({self.total_size},); got {array.shape}."
            )
        return array

    def split(self, value: ArrayLike, /) -> tuple[Array, ...]:
        flat = self._flat(value, "product state")
        return tuple(
            flat[left:right].reshape(block.shape)
            for block, left, right in zip(
                self.blocks,
                self.offsets[:-1],
                self.offsets[1:],
                strict=True,
            )
        )

    def combine(self, values: Sequence[ArrayLike], /) -> Array:
        values_ = tuple(values)
        if len(values_) != len(self.blocks):
            raise ValueError("Product-state values must match the block count.")
        flattened = []
        for block, value in zip(self.blocks, values_, strict=True):
            array = jnp.asarray(value)
            if array.shape != block.shape:
                raise ValueError(
                    f"Product block {block.block_id!r} must have shape {block.shape}."
                )
            flattened.append(array.reshape((-1,)))
        return jnp.concatenate(tuple(flattened))

    def contains(self, state: ArrayLike, /) -> Array:
        values = self.split(state)
        valid = jnp.asarray(True)
        for block, value in zip(self.blocks, values, strict=True):
            valid = valid & block.geometry.contains(value)
        return valid

    def project_tangent(self, state: ArrayLike, vector: ArrayLike, /) -> Array:
        states = self.split(state)
        vectors = self.split(vector)
        return self.combine(
            tuple(
                block.geometry.project_tangent(point, tangent)
                for block, point, tangent in zip(
                    self.blocks, states, vectors, strict=True
                )
            )
        )

    def to_local(self, state: ArrayLike, tangent: ArrayLike, /) -> Array:
        states = self.split(state)
        tangents = self.split(tangent)
        return self.combine(
            tuple(
                block.geometry.to_local(point, vector)
                for block, point, vector in zip(
                    self.blocks, states, tangents, strict=True
                )
            )
        )

    def from_local(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        states = self.split(state)
        locals_ = self.split(local_tangent)
        return self.combine(
            tuple(
                block.geometry.from_local(point, local)
                for block, point, local in zip(self.blocks, states, locals_, strict=True)
            )
        )

    def retract(self, state: ArrayLike, local_tangent: ArrayLike, /) -> Array:
        states = self.split(state)
        locals_ = self.split(local_tangent)
        return self.combine(
            tuple(
                block.geometry.retract(point, local)
                for block, point, local in zip(self.blocks, states, locals_, strict=True)
            )
        )

    def inverse_retract(self, state: ArrayLike, point: ArrayLike, /) -> Array:
        states = self.split(state)
        points = self.split(point)
        return self.combine(
            tuple(
                block.geometry.inverse_retract(left, right)
                for block, left, right in zip(self.blocks, states, points, strict=True)
            )
        )

    def pullback(
        self,
        state: ArrayLike,
        local_tangent: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        states = self.split(state)
        locals_ = self.split(local_tangent)
        tangents = self.split(tangent)
        return self.combine(
            tuple(
                block.geometry.pullback(point, local, vector)
                for block, point, local, vector in zip(
                    self.blocks, states, locals_, tangents, strict=True
                )
            )
        )


__all__ = ["ProductStateGeometry", "ProductStateGeometryBlock"]
